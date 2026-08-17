from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# DATASET
# ============================================================

DATA_PATH = DATA_DIR / "dataset.csv"


# ============================================================
# NLP PARAMETERS
# ============================================================

VOCAB_SIZE = 10000

MAX_SEQUENCE_LENGTH = 100

EMBEDDING_DIM = 128


# ============================================================
# TRAINING PARAMETERS
# ============================================================

TEST_SIZE = 0.20

VALIDATION_SPLIT = 0.20

RANDOM_STATE = 42

EPOCHS = 10

BATCH_SIZE = 64


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)