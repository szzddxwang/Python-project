from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

ROLLING_SALES_CSV = BASE_DIR / "data" / "rollingsales_queens_clean.csv"
GEOCODE_CACHE = BASE_DIR / "data" / "geocode_cache.parquet"
GEO_SEED_CSV  = BASE_DIR / "data" / "geocode_seed.csv"

GEOCODE_CITY_HINT = "Queens, NY"

APP_TITLE   = "Queens Housing Advisor "
DEFAULT_LAT = 40.728
DEFAULT_LON = -73.85

TREND_WEIGHT  = 0.444
RATIO_WEIGHT  = 0.333
AFFORD_WEIGHT = 0.222

FAST_START_STRICT = True

def ensure_data_exists():
    if not ROLLING_SALES_CSV.exists():
        raise FileNotFoundError(
            "Missing data/rollingsales_queens_clean.csv in ./data. "
            "Put your CSV here (or a same-named .parquet/.feather)."
        )
