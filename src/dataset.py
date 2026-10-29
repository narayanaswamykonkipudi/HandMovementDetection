"""Dataset validation, leakage-safe splitting, and processed-split I/O.

The invariants enforced here matter more than any individual model:

* every sample has an explicit subject and recording session;
* a recording session is never divided between train/validation/test;
* the held-out test split is created once and never used for tuning;
* invalid rows fail loudly instead of being silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    COLLECTION_LABELS,
    CSV_COLUMNS,
    FEATURE_DIM,
    HANDEDNESS_VALUES,
    RAW_COORD_COLUMNS,
    SOURCE_VALUES,
    STATIC_ASL_LETTERS,
    UNKNOWN_LABEL,
)


class DatasetError(ValueError):
    """Raised when a dataset cannot be used safely."""


@dataclass
class ValidationReport:
    n_samples: int = 0
    labels: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    subjects: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    sources: dict[str, int] = field(default_factory=dict)
    dataset_kind: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "labels": self.labels,
            "class_counts": self.class_counts,
            "subjects": self.subjects,
            "sessions": self.sessions,
            "sources": self.sources,
            "dataset_kind": self.dataset_kind,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_kind_from_sources(sources: set[str]) -> str:
    if sources == {"synthetic"}:
        return "synthetic"
    if "synthetic" in sources:
        return "mixed"
    return "real"


def group_ids(frame: pd.DataFrame) -> pd.Series:
    return frame["subject_id"].astype(str) + "::" + frame["session_id"].astype(str)


def validate_dataframe(
    frame: pd.DataFrame, *, for_split: bool = False
) -> ValidationReport:
    """Validate schema and values without mutating the data.

    ``for_split`` relaxes requirements that apply only to a complete dataset
    (for example, a test split may legitimately contain a single session).
    """

    report = ValidationReport()
    if frame.empty:
        report.errors.append("Dataset is empty.")
        return report

    missing_columns = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing_columns:
        report.errors.append(f"Missing required columns: {missing_columns}")
        return report

    report.n_samples = int(len(frame))
    report.labels = sorted(frame["label"].dropna().astype(str).unique().tolist())
    report.class_counts = {
        str(key): int(value)
        for key, value in frame["label"].astype(str).value_counts().sort_index().items()
    }
    report.subjects = sorted(frame["subject_id"].dropna().astype(str).unique().tolist())
    report.sessions = sorted(
        (
            frame["subject_id"].astype(str) + "/" + frame["session_id"].astype(str)
        ).unique().tolist()
    )
    report.sources = {
        str(key): int(value)
        for key, value in frame["source"].astype(str).value_counts().sort_index().items()
    }
    report.dataset_kind = dataset_kind_from_sources(set(report.sources))

    for column in (
        "sample_id",
        "label",
        "subject_id",
        "session_id",
        "source",
        "handedness",
    ):
        blanks = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
        if blanks.any():
            report.errors.append(
                f"Column '{column}' has {int(blanks.sum())} blank value(s)."
            )

    invalid_labels = sorted(set(report.labels) - set(COLLECTION_LABELS))
    if invalid_labels:
        report.errors.append(
            f"Unsupported label(s): {invalid_labels}. Static ASL letters are "
            f"{''.join(STATIC_ASL_LETTERS)}; use '{UNKNOWN_LABEL}' for rejection data."
        )

    duplicate_ids = frame["sample_id"].astype(str).duplicated(keep=False)
    if duplicate_ids.any():
        examples = frame.loc[duplicate_ids, "sample_id"].astype(str).head(5).tolist()
        report.errors.append(f"Duplicate sample_id values, e.g. {examples}.")

    invalid_sources = sorted(set(report.sources) - set(SOURCE_VALUES))
    if invalid_sources:
        report.errors.append(
            f"Invalid source value(s): {invalid_sources}. Expected {SOURCE_VALUES}."
        )

    invalid_handedness = sorted(
        set(frame["handedness"].dropna().astype(str)) - set(HANDEDNESS_VALUES)
    )
    if invalid_handedness:
        report.errors.append(
            f"Invalid handedness value(s): {invalid_handedness}. "
            f"Expected {HANDEDNESS_VALUES}."
        )

    try:
        scores = pd.to_numeric(frame["hand_score"], errors="raise").to_numpy(float)
    except (TypeError, ValueError):
        report.errors.append("hand_score contains non-numeric values.")
        scores = np.array([], dtype=float)
    if scores.size:
        if not np.all(np.isfinite(scores)):
            report.errors.append("hand_score contains NaN or infinite values.")
        if np.any((scores < 0.0) | (scores > 1.0)):
            report.errors.append("hand_score must be between 0 and 1.")
        low_scores = int((scores < 0.5).sum())
        if low_scores:
            report.warnings.append(
                f"{low_scores} sample(s) have hand_score < 0.5; inspect them before training."
            )

    try:
        coords = frame[list(RAW_COORD_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        report.errors.append("Landmark coordinate columns contain non-numeric values.")
        coords = np.empty((0, FEATURE_DIM), dtype=float)
    if coords.size:
        if not np.all(np.isfinite(coords)):
            bad = int((~np.isfinite(coords)).any(axis=1).sum())
            report.errors.append(f"{bad} row(s) contain NaN/infinite landmarks.")
        zero_rows = int((np.abs(coords).sum(axis=1) < 1e-9).sum())
        if zero_rows:
            report.errors.append(
                f"{zero_rows} row(s) have all-zero landmarks, which indicates corrupt data."
            )

    known_counts = {
        label: count
        for label, count in report.class_counts.items()
        if label != UNKNOWN_LABEL
    }
    if not known_counts and not for_split:
        report.errors.append(f"No known sign labels found (only {UNKNOWN_LABEL}).")
    elif known_counts:
        tiny = {label: count for label, count in known_counts.items() if count < 30}
        if tiny:
            report.warnings.append(
                f"Classes with fewer than 30 samples: {tiny}. Collect more sessions."
            )
        if len(known_counts) < 2 and not for_split:
            report.warnings.append(
                "Only one known class is present. A classifier needs at least two."
            )
        values = list(known_counts.values())
        if values and max(values) / max(min(values), 1) > 3.0:
            report.warnings.append(
                "Class imbalance is greater than 3:1 among known labels. "
                "Use class weights and report per-class metrics."
            )

    if len(report.subjects) < 2 and not for_split:
        report.warnings.append(
            "Fewer than two subjects. Unseen-person generalisation cannot be estimated."
        )

    group_series = group_ids(frame)
    if not for_split and group_series.nunique() < 3:
        report.errors.append(
            "At least three distinct subject/session groups are required for a "
            "train/validation/test split without leakage."
        )

    groups_per_label = (
        frame.assign(_group=group_series)
        .groupby("label", observed=True)["_group"]
        .nunique()
        .to_dict()
    )
    single_group_labels = [
        label
        for label, count in groups_per_label.items()
        if label != UNKNOWN_LABEL and int(count) < 2
    ]
    if single_group_labels and not for_split:
        report.warnings.append(
            f"These labels come from only one recording session: {single_group_labels}. "
            "They cannot be validated on unseen groups."
        )

    if UNKNOWN_LABEL not in report.class_counts and not for_split:
        report.warnings.append(
            "No '_unknown' rejection samples. Confidence thresholds will use a "
            "heuristic/validation-only fallback and unknown-sign rejection is unmeasured."
        )

    return report


def load_dataset(path: Path | str) -> tuple[pd.DataFrame, ValidationReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Collect data with collect_data.py first."
        )
    frame = pd.read_csv(path, dtype={"label": str, "subject_id": str, "session_id": str})
    report = validate_dataframe(frame)
    if report.errors:
        raise DatasetError(
            "Dataset validation failed:\n- " + "\n- ".join(report.errors)
        )
    return frame, report


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_samples": int(len(frame)),
        "n_groups": int(group_ids(frame).nunique()) if len(frame) else 0,
        "groups": sorted(group_ids(frame).unique().tolist()) if len(frame) else [],
        "samples_per_label": {
            str(key): int(value)
            for key, value in frame["label"].astype(str).value_counts().sort_index().items()
        },
        "subjects": sorted(frame["subject_id"].astype(str).unique().tolist()),
    }


def split_dataset(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    test_subjects: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Create train/validation/test splits at whole-session granularity.

    If ``test_subjects`` is supplied, all sessions belonging to those subjects
    are placed in the test split, giving a genuine unseen-person evaluation.
    Otherwise, stratified group folds are used: fold 0 becomes test, fold 1
    validation, and the remaining folds training.
    """

    if frame.empty:
        raise DatasetError("Cannot split an empty dataset.")

    frame = frame.copy()
    frame["_group"] = group_ids(frame)
    labels = frame["label"].astype(str)
    groups = frame["_group"]
    n_groups = int(groups.nunique())
    if n_groups < 3:
        raise DatasetError(
            f"Need at least 3 subject/session groups for train/val/test, found {n_groups}."
        )

    warnings: list[str] = []

    test_subjects = [str(subject) for subject in (test_subjects or []) if str(subject)]
    if test_subjects:
        unknown_subjects = sorted(set(test_subjects) - set(frame["subject_id"].astype(str)))
        if unknown_subjects:
            raise DatasetError(
                f"Requested test subject(s) not found: {unknown_subjects}. "
                f"Available: {sorted(frame['subject_id'].astype(str).unique().tolist())}"
            )
        test_mask = frame["subject_id"].astype(str).isin(test_subjects)
        test = frame[test_mask].copy()
        remaining = frame[~test_mask].copy()
        if remaining["_group"].nunique() < 2:
            raise DatasetError(
                "Not enough non-test groups remain to create train and validation splits."
            )
        n_folds = min(4, int(remaining["_group"].nunique()), int(remaining["label"].value_counts().min()))
        n_folds = max(2, n_folds)
        fold_splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        folds = list(
            fold_splitter.split(
                remaining,
                y=remaining["label"].astype(str),
                groups=remaining["_group"],
            )
        )
        val_idx = folds[0][1]
        val = remaining.iloc[val_idx].copy()
        train = remaining.drop(remaining.index[val_idx]).copy()
        method = f"subject_holdout + StratifiedGroupKFold(n_splits={n_folds}) for train/val"
    else:
        # ``_unknown`` is optional rejection data and is often much rarer than
        # the sign classes.  A handful of unknown rows must not make the whole
        # train/validation/test split impossible; the group splitter will place
        # their group in one fold.
        known_values = labels[labels.astype(str) != UNKNOWN_LABEL]
        fold_basis = known_values if not known_values.empty else labels
        min_class_samples = int(fold_basis.value_counts().min())
        n_folds = min(5, n_groups, min_class_samples)
        if n_folds < 3:
            raise DatasetError(
                "Each class needs at least 3 samples and 3 sessions are required for a "
                "leakage-free three-way split. Collect more sessions/subjects."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        folds = list(splitter.split(frame, y=labels, groups=groups))
        test_idx = folds[0][1]
        val_idx = folds[1][1]
        test = frame.iloc[test_idx].copy()
        val = frame.iloc[val_idx].copy()
        held_out = np.concatenate([test_idx, val_idx])
        train = frame.drop(frame.index[held_out]).copy()
        method = f"StratifiedGroupKFold(n_splits={n_folds}); fold 0=test, fold 1=validation"

    if len(train) + len(val) + len(test) != len(frame):  # pragma: no cover - defensive
        warnings.append("Split sizes do not add up to the dataset size.")

    # Explicit leakage assertions. These are cheap and catch future regressions.
    split_groups = {
        name: set(group_ids(part)) for name, part in (("train", train), ("val", val), ("test", test))
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise DatasetError(
                f"Leakage detected: groups shared by {left} and {right}: {sorted(overlap)}"
            )
    id_sets = {
        name: set(part["sample_id"].astype(str)) for name, part in (("train", train), ("val", val), ("test", test))
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise DatasetError(f"Sample IDs shared by {left} and {right}: {len(overlap)}")

    known_labels = set(STATIC_ASL_LETTERS)
    train_labels = set(train.loc[train["label"] != UNKNOWN_LABEL, "label"].astype(str))
    if train_labels != known_labels:
        missing = sorted(known_labels - train_labels)
        extra = sorted(train_labels - known_labels)
        raise DatasetError(
            "Training split must contain every static ASL letter. "
            f"Missing: {missing}; unexpected: {extra}. Collect more sessions per label."
        )

    for name, part in (("validation", val), ("test", test)):
        missing = sorted(known_labels - set(part["label"].astype(str)))
        if missing:
            warnings.append(
                f"{name} split is missing class(es) {missing}; per-class metrics for "
                "those letters are not estimable."
            )
        if UNKNOWN_LABEL not in set(part["label"].astype(str)):
            warnings.append(
                f"{name} split has no '_unknown' samples; unknown-sign rejection "
                "cannot be measured on this split."
            )

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise DatasetError("One of the splits is empty. Collect more sessions.")

    for _name, part in (("train", train), ("val", val), ("test", test)):
        part.drop(columns=["_group"], inplace=True)

    info = {
        "method": method,
        "seed": seed,
        "test_subjects": test_subjects,
        "n_folds": int(n_folds),
        "warnings": warnings,
        "test_contains_only_unseen_subjects": bool(
            test_subjects
            and not set(test["subject_id"].astype(str)) & set(train["subject_id"].astype(str))
        ),
        "splits": {
            "train": _split_summary(train),
            "val": _split_summary(val),
            "test": _split_summary(test),
        },
    }
    return {"train": train, "val": val, "test": test}, info


def write_processed_splits(
    splits: dict[str, pd.DataFrame],
    info: dict[str, Any],
    source_path: Path | str,
    report: ValidationReport,
    output_dir: Path | str,
) -> Path:
    """Write CSV splits plus a manifest recording provenance and leakage checks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path).resolve()

    for name in ("train", "val", "test"):
        if name not in splits:
            raise DatasetError(f"Missing split '{name}'.")
        splits[name][list(CSV_COLUMNS)].to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "source_csv": str(source_path),
        "source_sha256": sha256_file(source_path),
        "dataset": report.to_dict(),
        "split": info,
        "feature_spec": {
            "landmark_count": 21,
            "feature_dim": FEATURE_DIM,
            "coordinate_columns": list(RAW_COORD_COLUMNS),
            "normalization": "wrist translation + mean palm-base scale; orientation preserved",
        },
        "notes": [
            "Groups are subject_id::session_id. No group appears in more than one split.",
            "The test split must not be used for feature selection, hyperparameters, or thresholds.",
            "dataset_kind 'synthetic' means the data is a plumbing test, not real sign data.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_groups = set(info["splits"]["train"]["groups"])
    val_groups = set(info["splits"]["val"]["groups"])
    test_groups = set(info["splits"]["test"]["groups"])
    if (train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups):
        raise DatasetError("Internal error: manifests contain overlapping groups.")

    return manifest_path


def load_processed_splits(
    processed_dir: Path | str,
    *,
    names: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load processed splits and their manifest.

    ``names`` lets training read only train/validation files, so the test CSV is
    not even parsed during model development.
    """

    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run preprocess_data.py before training/evaluation."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, pd.DataFrame] = {}
    for name in names:
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split name: {name}")
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = pd.read_csv(path, dtype={"label": str, "subject_id": str, "session_id": str})
    for name, frame in splits.items():
        report = validate_dataframe(frame, for_split=True)
        # Split files intentionally contain only the classes present in that
        # split, so a missing class warning is useful but not fatal.
        if report.errors:
            raise DatasetError(
                f"Processed split '{name}' is invalid:\n- " + "\n- ".join(report.errors)
            )
    return splits, manifest
