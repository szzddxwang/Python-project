# tools/merge_cache_into_csv.py — join geocode_cache into CSV as LAT/LON
from pathlib import Path
import pandas as pd
from data_io import _norm_addr
from config import ROLLING_SALES_CSV, GEOCODE_CACHE

csv = Path(ROLLING_SALES_CSV)
cache = Path(GEOCODE_CACHE)
df = pd.read_csv(csv)
df.columns = [c.strip().upper() for c in df.columns]
df["ZIP CODE"] = df["ZIP CODE"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
key = (df["ADDRESS"].astype(str).str.title() + ", " + df["ZIP CODE"]).apply(_norm_addr)

c = pd.read_parquet(cache)
key_cache = c["full_address"].astype(str).apply(_norm_addr)
ll = dict(zip(key_cache, zip(c["lat"], c["lon"])))

df["LAT"] = key.map(lambda k: ll.get(k, (float("nan"), float("nan")))[0])
df["LON"] = key.map(lambda k: ll.get(k, (float("nan"), float("nan")))[1])

out = csv.with_name(csv.stem + "_with_ll.csv")
df.to_csv(out, index=False)
print("Wrote:", out)
