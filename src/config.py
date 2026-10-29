"""Project-wide paths, schema constants, and scope definitions.

All modules import the constants from here so that the CSV schema, feature
dimension, label set, and file locations cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DATA_CSV = RAW_DIR / "landmarks.csv"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
SAMPLE_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
DEMO_CLASSIFIER_PATH = MODELS_DIR / "demo_classifier.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Linguistic scope
# ---------------------------------------------------------------------------
# American Sign Language (ASL) fingerspelling, static handshapes only.
# J and Z require motion and are deliberately excluded: a single-frame
# classifier cannot represent them correctly.  This is not a sign-language
# translator and does not model grammar, facial expression, or dynamic signs.
STATIC_ASL_LETTERS: tuple[str, ...] = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
DYNAMIC_ASL_LETTERS: tuple[str, ...] = ("J", "Z")
UNKNOWN_LABEL = "_unknown"
COLLECTION_LABELS: tuple[str, ...] = STATIC_ASL_LETTERS + (UNKNOWN_LABEL,)
SOURCE_VALUES: tuple[str, ...] = ("webcam", "synthetic", "imported")
HANDEDNESS_VALUES: tuple[str, ...] = ("Left", "Right", "Unknown")

# ---------------------------------------------------------------------------
# Landmark schema
# ---------------------------------------------------------------------------
LANDMARK_COUNT = 21
FEATURE_DIM = LANDMARK_COUNT * 3  # x, y, z for every landmark
MP_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

RAW_COORD_COLUMNS: tuple[str, ...] = tuple(
    f"{axis}{index}" for index in range(LANDMARK_COUNT) for axis in ("x", "y", "z")
)
METADATA_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "label",
    "subject_id",
    "session_id",
    "source",
    "handedness",
    "hand_score",
    "timestamp",
)
CSV_COLUMNS: tuple[str, ...] = METADATA_COLUMNS + RAW_COORD_COLUMNS

# Landmark indices used by the palm-scale normalisation and overlays.
WRIST_INDEX = 0
PALM_BASE_INDICES = (5, 9, 13, 17)  # index/middle/ring/pinky MCP joints

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_SMOOTHING_WINDOW = 8
DEFAULT_SMOOTHING_MIN_VOTES = 5
DEFAULT_SMOOTHING_MIN_RATIO = 0.60
DEFAULT_RELEASE_FRAMES = 8
MAX_TEXT_LENGTH = 280

# Confidence/margin defaults used only before validation calibration.
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_MARGIN = 0.15

MODEL_FORMAT_VERSION = 2


def ensure_directories() -> None:
    """Create the writable directories used by the CLI scripts."""

    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        DEMO_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
