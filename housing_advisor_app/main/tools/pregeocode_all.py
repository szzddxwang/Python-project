# pregeocode_all.py — offline geocoding with progress and periodic flush

import os, re, time, sys, signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from config import ROLLING_SALES_CSV, GEOCODE_CACHE, GEO_SEED_CSV, GEOCODE_CITY_HINT

try:
    from geopy.geocoders import GoogleV3, MapBox, Nominatim
except Exception as e:
    print("Missing dependency 'geopy'. Install via: pip install geopy", file=sys.stderr)
    raise

APT_PAT = re.compile(r"\b(apt|unit|#)\s*[\w\-]+", re.IGNORECASE)
def _norm_addr(s: str) -> str:
    x = str(s)
    x = APT_PAT.sub("", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x.title()

def load_base() -> pd.DataFrame:
    p = Path(ROLLING_SALES_CSV)
    if p.with_suffix(".parquet").exists():
        df = pd.read_parquet(p.with_suffix(".parquet"))
    elif p.with_suffix(".feather").exists():
        df = pd.read_feather(p.with_suffix(".feather"))
    else:
        df = pd.read_csv(p)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.dropna(subset=["ADDRESS","ZIP CODE"])
    df["ZIP CODE"] = df["ZIP CODE"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
    base = pd.DataFrame({"address": df["ADDRESS"].astype(str).str.title(), "postcode": df["ZIP CODE"]})
    base["full_address"] = (base["address"] + ", " + base["postcode"]).apply(_norm_addr)
    return base.drop_duplicates("full_address")

def choose_provider():
    prov_env = os.getenv("GEOCODER_PROVIDER", "").strip().lower()
    gkey = os.getenv("GOOGLE_API_KEY", "").strip()
    mkey = os.getenv("MAPBOX_TOKEN", "").strip()

    if prov_env in ("google", "") and gkey:
        workers = int(os.getenv("GOOGLE_MAX_WORKERS", "12"))
        qps     = int(os.getenv("GOOGLE_QPS", "12"))
        return "google", GoogleV3(api_key=gkey, timeout=10), workers, qps

    if prov_env in ("mapbox",) and mkey:
        workers = int(os.getenv("MAPBOX_MAX_WORKERS", "12"))
        qps     = int(os.getenv("MAPBOX_QPS", "12"))
        return "mapbox", MapBox(api_key=mkey, timeout=10), workers, qps

    # fallback
    return "nominatim", Nominatim(user_agent="housing_advisor_pregeocode"), 1, 1

def rate_batches(items: List[str], qps: int) -> List[List[str]]:
    qps = max(1, qps)
    out, b = [], []
    for it in items:
        b.append(it)
        if len(b) >= qps:
            out.append(b); b=[]
    if b: out.append(b)
    return out

def save_cache(cache_map: Dict[str, Tuple[float,float]]):
    Path(GEOCODE_CACHE).parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{"full_address": k, "lat": v[0], "lon": v[1]} for k, v in cache_map.items()])
    out.drop_duplicates(subset=["full_address"], keep="last", inplace=True)
    out.to_parquet(GEOCODE_CACHE, index=False)
    print(f"[flush] wrote cache → {GEOCODE_CACHE} | rows={len(out)}")

def main():
    base = load_base()
    key_to_query = {fa: _norm_addr(f"{a}, {GEOCODE_CITY_HINT} {z}") for a, z, fa in zip(base["address"], base["postcode"], base["full_address"])}

    # Load existing cache
    cache: Dict[str, Tuple[float, float]] = {}
    if Path(GEOCODE_CACHE).exists():
        try:
            c = pd.read_parquet(GEOCODE_CACHE)
            cache.update({fa:(float(lat),float(lon)) for fa,lat,lon in zip(c["full_address"],c["lat"],c["lon"])})
        except Exception as e:
            print(f"[warn] failed to read existing cache {GEOCODE_CACHE}: {e}")
    if Path(GEO_SEED_CSV).exists():
        try:
            s = pd.read_csv(GEO_SEED_CSV)
            if {"full_address","lat","lon"}.issubset(set(s.columns)):
                s["full_address"] = s["full_address"].astype(str).apply(_norm_addr)
                cache.update({fa:(float(lat),float(lon)) for fa,lat,lon in zip(s["full_address"],s["lat"],s["lon"])})
        except Exception as e:
            print(f"[warn] failed to read seed {GEO_SEED_CSV}: {e}")

    pending = [fa for fa in base["full_address"] if fa not in cache]
    name, provider, max_workers, qps = choose_provider()
    print(f"Provider: {name} | pending: {len(pending)} | qps={qps} workers={max_workers}")

    if not pending:
        print("Nothing to geocode. Cache already full.")
        save_cache(cache)
        return

    # Periodically save parameters to disk
    FLUSH_EVERY_BATCHES = int(os.getenv("FLUSH_EVERY_BATCHES", "20"))
    FLUSH_EVERY_SECONDS = int(os.getenv("FLUSH_EVERY_SECONDS", "30"))
    last_flush = time.time()

    queries = [key_to_query[fa] for fa in pending]
    total = len(queries)
    done = 0
    partial_cache = dict(cache)

    # Supports Ctrl+C: save to disk before exiting, allowing safe resumption of the run
    def handle_sigint(sig, frame):
        print("\n[signal] KeyboardInterrupt received. Flushing current progress to disk...")
        save_cache(partial_cache)
        print("[signal] Safe to re-run later; will resume from cache.")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    def geocode_one(q: str):
        try:
            if name == "nominatim":
                loc = provider.geocode(q, timeout=10)
            else:
                loc = provider.geocode(q)
            if not loc:
                return (float("nan"), float("nan"))
            return (float(loc.latitude), float(loc.longitude))
        except Exception:
            return (float("nan"), float("nan"))

    inv = {v: k for k, v in key_to_query.items()}
    batches = rate_batches(queries, qps=qps)

    for b in batches:
        # batch of requests
        if name == "nominatim":
            res_local: Dict[str, Tuple[float,float]] = {}
            for q in b:
                res_local[q] = geocode_one(q)
            time.sleep(1)   # 礼貌限速
        else:
            res_local = {}
            with ThreadPoolExecutor(max_workers=min(max_workers, len(b))) as ex:
                futs = {ex.submit(geocode_one, q): q for q in b}
                for f in as_completed(futs):
                    q = futs[f]
                    try: res_local[q] = f.result()
                    except Exception: res_local[q] = (float("nan"), float("nan"))
            time.sleep(1)   # 批间节流

        # merge results
        for q, (lat, lon) in res_local.items():
            fa = inv.get(q)
            if fa:
                partial_cache[fa] = (lat, lon)

        done += len(b)
        print(f"[progress] {done}/{total} geocoded")

        # the batch count or time threshold is reached and save to disk
        now = time.time()
        if (done % (FLUSH_EVERY_BATCHES * max(1, len(b))) == 0) or (now - last_flush >= FLUSH_EVERY_SECONDS):
            save_cache(partial_cache)
            last_flush = now

    # save to disk
    save_cache(partial_cache)
    print(f"Wrote cache: {GEOCODE_CACHE} | rows={len(partial_cache)}")

if __name__ == "__main__":
    main()
