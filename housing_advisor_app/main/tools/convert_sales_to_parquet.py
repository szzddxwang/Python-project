# tools/convert_sales_to_parquet.py — CSV → Parquet/Feather
from pathlib import Path
import sys, pandas as pd

BASE = Path(__file__).resolve().parents[1]
CSV  = BASE / "data" / "rollingsales_queens_clean.csv"

def main(fmt="parquet"):
    if not CSV.exists():
        print(f"CSV not found: {CSV}"); sys.exit(1)
    df = pd.read_csv(CSV, dtype=str)
    if fmt.lower() == "parquet":
        out = CSV.with_suffix(".parquet")
        df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    else:
        out = CSV.with_suffix(".feather")
        df.to_feather(out)
    print("Wrote:", out)

if __name__ == "__main__":
    fmt = sys.argv[1] if len(sys.argv) > 1 else "parquet"
    main(fmt)
