#!/usr/bin/env python
"""Convert a class-folder image dataset into landmark rows.

The Kaggle ASL Alphabet layout is often nested one level deeper than expected::

    Dataset_kaggle/
        asl_alphabet_train/
            asl_alphabet_train/
                A/ A1.jpg ...
                B/ ...
                ...

Pass the outer directory (or any parent up to three levels above the class
folders) and the script discovers the actual class root automatically.

Example
-------
    python scripts/import_image_dataset.py \
        --root Dataset_kaggle/asl_alphabet_train \
        --max-per-class 1500 \
        --output data/imported/landmarks.csv

Behaviour
---------
* Keeps the 24 static ASL letters A-Z except J/Z.
* Maps ``nothing`` to ``_unknown`` and skips ``J``, ``Z``, ``del`` and ``space``.
* Deterministically samples at most ``--max-per-class`` images per class.
* Runs MediaPipe locally and stores landmarks only; no images are uploaded or
  copied into the project.
* Groups samples into deterministic pseudo-sessions.  This limits near-duplicate
  leakage between splits but is **not** a subject-disjoint evaluation.
* Can resume after an interruption: ``--resume`` skips images whose deterministic
  sample id is already present in the output CSV.

Important limitations
---------------------
* Public image folders usually do not provide signer/session metadata.  Do not
  claim unseen-signer accuracy from the resulting split.
* J and Z are skipped because static images cannot represent dynamic letters.
* Verify the dataset's licence before use and do not redistribute its images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collector import append_samples, count_samples, make_sample_row
from src.config import HAND_MODEL_PATH, STATIC_ASL_LETTERS, UNKNOWN_LABEL

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISCOVERY_DEPTH = 3

# Folder names in the source dataset that are deliberately not imported.
SKIPPED_FOLDERS = {
    "J": "dynamic letter (requires motion)",
    "Z": "dynamic letter (requires motion)",
    "DEL": "delete command, not a static letter",
    "SPACE": "space command, not a static letter",
}
# ``nothing`` is useful as rejection training data.
LABEL_ALIASES = {"NOTHING": UNKNOWN_LABEL}


class ImportConfigError(RuntimeError):
    """Raised for recoverable import configuration errors."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory (class folders may be nested up to 3 levels below).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "imported" / "landmarks.csv",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1500,
        help="Maximum number of source images sampled per class (default: 1500).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--session-size",
        type=int,
        default=100,
        help="Images per pseudo-session; spreads each class over several groups.",
    )
    parser.add_argument("--subject-id", default="kaggle", help="Subject id written to CSV.")
    parser.add_argument(
        "--min-hand-score",
        type=float,
        default=0.5,
        help="Discard saved samples below this MediaPipe presence score (0-1).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.1,
        help=(
            "MediaPipe proposal threshold. Keep this low so closed-fist letters "
            "(for example M/N) are proposed, then filter with --min-hand-score."
        ),
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=0.1,
        help="MediaPipe presence threshold used while proposing a hand.",
    )
    parser.add_argument(
        "--include-flipped",
        action="store_true",
        help=(
            "Also import a horizontally flipped variant of every image. Not needed "
            "when training with the project's mirror augmentation, but useful for a "
            "direct mirror-invariance check."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild output.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output CSV instead of refusing to overwrite it.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print discovered classes and image counts, then exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery and deterministic selection
# ---------------------------------------------------------------------------
def _is_class_folder(path: Path) -> bool:
    return path.is_dir() and path.name.strip().upper() in {
        "A",
        "B",
        "C",
        "DEL",
        "NOTHING",
        "SPACE",
    }


def discover_class_root(root: Path) -> Path:
    """Return the deepest directory whose immediate children are class folders."""

    root = root.resolve()
    if not root.is_dir():
        raise ImportConfigError(f"Dataset root is not a directory: {root}")

    if any(_is_class_folder(child) for child in root.iterdir()):
        return root

    # Breadth-first search so a shallow, expected layout wins over an unrelated
    # nested duplicate.
    frontier: list[tuple[Path, int]] = [(root, 0)]
    candidates: list[Path] = []
    while frontier:
        directory, depth = frontier.pop(0)
        if depth >= MAX_DISCOVERY_DEPTH:
            continue
        children = sorted((child for child in directory.iterdir() if child.is_dir()), key=str)
        if any(_is_class_folder(child) for child in children):
            candidates.append(directory)
        for child in children:
            frontier.append((child, depth + 1))
    if not candidates:
        raise ImportConfigError(
            "Could not find a directory whose immediate children are class folders "
            f"within {MAX_DISCOVERY_DEPTH} levels of {root}."
        )
    # Prefer the candidate with the most recognised class folders; ties go to the
    # shallowest path.
    return max(
        candidates,
        key=lambda path: (
            sum(1 for child in path.iterdir() if _is_class_folder(child)),
            -len(path.parts),
        ),
    )


def folder_label(name: str) -> str | None:
    """Map a source folder to a project label, or ``None`` to skip it."""

    key = name.strip().upper()
    if key in SKIPPED_FOLDERS:
        return None
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    if key in STATIC_ASL_LETTERS:
        return key
    return None


def discover_classes(class_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Return ``{label: folder}`` plus a human-readable list of skipped folders."""

    found: dict[str, Path] = {}
    skipped: list[str] = []
    for path in sorted(class_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        label = folder_label(path.name)
        if label is None:
            key = path.name.strip().upper()
            if key in SKIPPED_FOLDERS:
                skipped.append(f"{path.name} ({SKIPPED_FOLDERS[key]})")
            else:
                skipped.append(f"{path.name} (not a supported class)")
            continue
        if label in found:
            raise ImportConfigError(
                f"Two folders map to label {label!r}: {found[label].name} and {path.name}."
            )
        found[label] = path
    return found, skipped


def image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def select_images(paths: Iterable[Path], *, max_per_class: int, seed: int, label: str) -> list[Path]:
    selected = list(paths)
    # Seed per class so adding/removing a class does not change other classes.
    random.Random(f"{seed}:{label}").shuffle(selected)
    # Restore source order after sampling.  Pseudo-sessions are then contiguous
    # blocks of source filenames, which keeps near-duplicate frames in the same
    # group and reduces leakage between train/validation/test.
    return sorted(selected[:max_per_class])


def deterministic_sample_id(source_root: Path, image_path: Path, variant: str) -> str:
    try:
        relative = image_path.resolve().relative_to(source_root).as_posix()
    except ValueError:
        relative = image_path.resolve().as_posix()
    return hashlib.sha1(f"{relative}|{variant}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Resume state and progress display
# ---------------------------------------------------------------------------
def existing_sample_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return set()
        return {str(row.get("sample_id", "")) for row in reader if row.get("sample_id")}


def _stats_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.import_stats.json")


def _empty_stats() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": {},
        "totals": {"attempted": 0, "usable": 0, "skipped": 0, "resumed": 0},
    }


def load_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_stats()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_stats()
    if not isinstance(payload, dict) or not isinstance(payload.get("classes"), dict):
        return _empty_stats()
    return payload


def save_stats(path: Path, stats: dict[str, Any]) -> None:
    payload = dict(stats)
    payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class ProgressBar:
    """Small dependency-free progress bar with a rolling ETA."""

    def __init__(self, total: int, prefix: str, width: int = 28) -> None:
        self.total = max(total, 0)
        self.prefix = prefix
        self.width = width
        self.done = 0
        self.started = time.perf_counter()
        self._last_render = 0.0

    def update(self, increment: int = 1, *, force: bool = False) -> None:
        self.done += increment
        now = time.perf_counter()
        if not force and now - self._last_render < 0.15 and self.done < self.total:
            return
        self._last_render = now
        ratio = 1.0 if self.total == 0 else min(self.done / self.total, 1.0)
        filled = int(round(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(now - self.started, 1e-9)
        rate = self.done / elapsed
        remaining = max(self.total - self.done, 0)
        eta = remaining / rate if rate > 0 else 0.0
        print(
            f"\r{self.prefix} [{bar}] {self.done}/{self.total} "
            f"({ratio * 100:5.1f}%) {rate:5.1f} img/s ETA {eta:4.0f}s",
            end="",
            flush=True,
        )
        if self.done >= self.total:
            print()

    def close(self) -> None:
        if self.done < self.total:
            self.update(force=True)
            print()


# ---------------------------------------------------------------------------
# Main import loop
# ---------------------------------------------------------------------------
def flush_batch(
    output: Path, rows: list[dict[str, Any]], seen_ids: set[str]
) -> int:
    """Append one batch and remember the ids so resume sees it immediately."""

    if not rows:
        return 0
    written = append_samples(output, rows)
    seen_ids.update(str(row["sample_id"]) for row in rows)
    rows.clear()
    return written


def _ensure_output_state(output: Path, *, overwrite: bool, resume: bool) -> set[str]:
    if output.exists() and overwrite:
        output.unlink()
        _stats_path(output).unlink(missing_ok=True)
    if output.exists() and not overwrite and not resume:
        raise ImportConfigError(
            f"Output already exists: {output}. Pass --resume to continue it or "
            "--overwrite to start again."
        )
    return existing_sample_ids(output) if output.exists() else set()


def main() -> int:
    args = parse_args()
    try:
        class_root = discover_class_root(args.root)
        classes, skipped_folders = discover_classes(class_root)
    except (ImportConfigError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not classes:
        print("ERROR: no supported static-letter class folders were found.", file=sys.stderr)
        return 2
    if args.max_per_class < 1 or args.session_size < 1:
        print("ERROR: --max-per-class and --session-size must be positive.", file=sys.stderr)
        return 2
    for name in (
        "min_hand_score",
        "min_detection_confidence",
        "min_presence_confidence",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            print(f"ERROR: --{name.replace('_', '-')} must be between 0 and 1.", file=sys.stderr)
            return 2
    if not HAND_MODEL_PATH.exists():
        print(
            f"ERROR: hand model missing at {HAND_MODEL_PATH}. "
            "Run: python scripts/download_models.py",
            file=sys.stderr,
        )
        return 2

    print(f"Dataset root: {args.root.resolve()}")
    print(f"Class root:   {class_root}")
    print(f"Output:       {args.output.resolve()}")
    print(f"Subject id:   {args.subject_id}")
    print(f"Sampling:     max {args.max_per_class}/class, seed={args.seed}, "
          f"pseudo-session size={args.session_size}")
    print()

    all_images = {label: image_files(folder) for label, folder in classes.items()}
    total_available = sum(len(paths) for paths in all_images.values())
    print(f"{'class':>9} {'available':>10} {'selected':>10}")
    for label in sorted(all_images, key=lambda item: (item == UNKNOWN_LABEL, item)):
        print(f"{label:>9} {len(all_images[label]):>10} "
              f"{min(len(all_images[label]), args.max_per_class):>10}")
    print(f"{'TOTAL':>9} {total_available:>10}")
    if skipped_folders:
        print("\nSkipped source folders:")
        for item in skipped_folders:
            print(f"  - {item}")
    print()

    if args.list_only:
        print("List-only mode: no landmarks were written.")
        return 0

    try:
        existing_ids = _ensure_output_state(
            args.output, overwrite=args.overwrite, resume=args.resume
        )
    except ImportConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stats_path = _stats_path(args.output)
    stats = load_stats(stats_path)
    stats["source_root"] = str(class_root)
    stats["output"] = str(args.output.resolve())
    stats["max_per_class"] = args.max_per_class
    stats["seed"] = args.seed
    stats["session_size"] = args.session_size
    stats["subject_id"] = args.subject_id
    stats["include_flipped"] = bool(args.include_flipped)

    print(
        f"Resuming from {len(existing_ids)} existing samples."
        if existing_ids
        else "Starting a fresh import.",
    )
    print("Licence reminder: verify the source dataset's terms. This script stores\n"
          "landmarks only and does not copy or redistribute images.\n")

    import cv2

    from src.hand_tracking import HandTracker, HandTrackerError

    tracker = HandTracker(
        running_mode="image",
        num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_detection_confidence,
    )
    variants = ("orig", "flip") if args.include_flipped else ("orig",)
    total_selected = 0
    total_new = 0
    started = time.perf_counter()
    tracker.open()
    try:
        for label in sorted(all_images, key=lambda item: (item == UNKNOWN_LABEL, item)):
            selected = select_images(
                all_images[label],
                max_per_class=args.max_per_class,
                seed=args.seed,
                label=label,
            )
            total_selected += len(selected) * len(variants)
            class_stats = stats["classes"].setdefault(
                label,
                {"selected": 0, "attempted": 0, "usable": 0, "skipped": 0, "resumed": 0},
            )
            class_stats["selected"] = len(selected) * len(variants)
            print(f"\n{label}: {len(selected)} source images, variants={','.join(variants)}")

            pending: list[tuple[Path, int, str, str]] = []
            for variant in variants:
                for index, image_path in enumerate(selected):
                    sample_id = deterministic_sample_id(class_root, image_path, variant)
                    if sample_id in existing_ids:
                        class_stats["resumed"] = int(class_stats.get("resumed", 0)) + 1
                        continue
                    pending.append((image_path, index, variant, sample_id))

            progress = ProgressBar(len(pending), prefix=f"  {label}")
            batch: list[dict[str, Any]] = []

            for image_path, index, variant, sample_id in pending:
                class_stats["attempted"] = int(class_stats.get("attempted", 0)) + 1
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    class_stats["skipped"] = int(class_stats.get("skipped", 0)) + 1
                    progress.update()
                    continue
                if variant == "flip":
                    frame = cv2.flip(frame, 1)
                try:
                    detections = tracker.detect(frame)
                except HandTrackerError as exc:
                    # A single malformed image should not destroy hours of work.
                    class_stats["skipped"] = int(class_stats.get("skipped", 0)) + 1
                    print(f"\n  warning: {image_path.name}: {exc}", file=sys.stderr)
                    progress.update()
                    continue
                if not detections or detections[0].score < args.min_hand_score:
                    class_stats["skipped"] = int(class_stats.get("skipped", 0)) + 1
                    progress.update()
                    continue
                primary = detections[0]
                row = make_sample_row(
                    primary.landmarks,
                    label=label,
                    subject_id=args.subject_id,
                    session_id=f"{label}-pseudo{index // args.session_size:03d}",
                    source="imported",
                    handedness=primary.handedness,
                    hand_score=primary.score,
                    sample_id=sample_id,
                )
                batch.append(row)
                class_stats["usable"] = int(class_stats.get("usable", 0)) + 1
                total_new += 1
                if len(batch) >= 25:
                    flush_batch(args.output, batch, existing_ids)
                progress.update()

            progress.close()
            flush_batch(args.output, batch, existing_ids)
            save_stats(stats_path, stats)

            attempted = int(class_stats.get("attempted", 0))
            usable = int(class_stats.get("usable", 0))
            skipped = int(class_stats.get("skipped", 0))
            resumed = int(class_stats.get("resumed", 0))
            rate = usable / attempted if attempted else None
            rate_text = f"{rate * 100:5.1f}%" if rate is not None else "  n/a "
            print(
                f"  {label}: usable {usable}/{attempted} processed "
                f"(detection {rate_text}), skipped {skipped}, resumed {resumed}"
            )
    finally:
        tracker.close()

    elapsed = time.perf_counter() - started
    counts = count_samples(args.output)
    stats["totals"] = {
        "selected": total_selected,
        "new_samples": total_new,
        "usable": sum(int(item.get("usable", 0)) for item in stats["classes"].values()),
        "skipped": sum(int(item.get("skipped", 0)) for item in stats["classes"].values()),
        "resumed": sum(int(item.get("resumed", 0)) for item in stats["classes"].values()),
        "elapsed_seconds": round(elapsed, 1),
    }
    save_stats(stats_path, stats)

    if not args.output.exists() or sum(counts.values()) == 0:
        print("ERROR: no usable samples were extracted.", file=sys.stderr)
        return 3

    print("\n" + "=" * 72)
    print(f"Wrote/kept {sum(counts.values())} samples in {args.output}")
    print(f"New this run: {total_new} | elapsed: {elapsed / 60:.1f} min")
    print(f"Per-class counts: {dict(sorted(counts.items()))}")
    print(f"Import stats: {stats_path}")
    print("\nDetection rates:")
    for label in sorted(stats["classes"], key=lambda item: (item == UNKNOWN_LABEL, item)):
        item = stats["classes"][label]
        attempted = int(item.get("attempted", 0))
        usable = int(item.get("usable", 0))
        rate = f"{usable / attempted * 100:5.1f}%" if attempted else "  n/a "
        print(
            f"  {label:>9}: {usable:>5} usable / {attempted:>5} processed "
            f"({rate}), skipped {int(item.get('skipped', 0)):>4}, "
            f"resumed {int(item.get('resumed', 0)):>5}"
        )
    print(
        "\nNext:\n"
        f"  python preprocess_data.py --input {args.output} --out data/processed_kaggle\n"
        f"  python train_model.py --processed data/processed_kaggle --model svm "
        "--out models/classifier_svm.joblib --no-tune\n"
        "\nThis split is pseudo-session grouped, not subject-disjoint. Do not claim "
        "unseen-signer accuracy from it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
