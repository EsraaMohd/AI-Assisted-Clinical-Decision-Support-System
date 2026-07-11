from pathlib import Path
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]    # parent[2]= AI_Assisted_..., parent[0]= utils
# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"