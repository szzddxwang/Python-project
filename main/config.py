from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

# Main sales data
ROLLING_SALES_CSV = BASE_DIR / "data" / "rollingsales_queens_clean.csv"

# Geocode cache
GEOCODE_CACHE = BASE_DIR / "data" / "geocode_cache"
GEO_SEED_CSV  = BASE_DIR / "data" / "geocode_seed.csv"

GEOCODE_CITY_HINT = "Queens, NY"

APP_TITLE   = "Queens Housing Advisor"
DEFAULT_LAT = 40.728
DEFAULT_LON = -73.85

TREND_WEIGHT  = 0.444
RATIO_WEIGHT  = 0.333
AFFORD_WEIGHT = 0.222

FAST_START_STRICT = True


def ensure_data_exists():
    """
    Basic check that the main sales file exists
    """
    csv_path = ROLLING_SALES_CSV
    parq_path = csv_path.with_suffix(".parquet")
    feat_path = csv_path.with_suffix(".feather")

    if csv_path.exists() or parq_path.exists() or feat_path.exists():
        return

    raise FileNotFoundError(
        "Missing data/rollingsales_queens_clean.(csv|parquet|feather) in ./data. "
        "Put your cleaned sales file here."
    )
