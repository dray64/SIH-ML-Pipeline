from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "D:\\flash-flood-early-warning\\data\\flash_flood_dataset_sensor_ready.csv"
CACHE_BUFFER_PATH = DATA_DIR / "cache" / "buffer_state.json"
MODEL_PATH = BASE_DIR / "models" / "D:\\flash-flood-early-warning\\models\\flash_flood_dual_stage_model.joblib"

# Hydrological Thresholds (in feet)
ACTION_STAGE_FT = 10.0
MAJOR_FLOOD_STAGE_FT = 17.0
ADVISORY_STAGE_FT = 6.0
CRITICAL_RISE_RATE_FT_HR = 3.0
WARNING_RISE_RATE_FT_HR = 1.5

# Buffer Configuration
MAX_BUFFER_HOURS = 30
HORIZONS = [1, 2, 3]