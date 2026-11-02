"""
Configuration and Path Management for StudyFlow.
Uses pathlib for OS-agnostic, dynamic path resolution.
this is to avoid the error running the model different in (windows, linux, mac) os 

what this does is built a common path way for all the os and make the file usable in every OS
"""

from pathlib import Path

# Project root (dynamically resolved)
BASE_DIR = Path(__file__).resolve().parents[2]  # or .parent.parent.parent
# Core Directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# File Paths
DATASET_PATH = DATA_DIR / "student-mat.csv"
DATABASE_PATH = DATA_DIR / "studyflow.db"
MODEL_PATH = MODELS_DIR / "best_student_model.joblib"
FEATURES_PATH = MODELS_DIR / "model_features.joblib"

# Educational Domain Constants
PASSING_GRADE = 10.0
MAX_GRADE = 20.0
MIN_GRADE = 0.0
DROPOUT_RISK_THRESHOLD = 9.0