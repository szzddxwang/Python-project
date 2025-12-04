from pathlib import Path
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ROLLING_SALES_CSV,
    GEOCODE_CACHE,
    GEO_SEED_CSV,
    FAST_START_STRICT,
)


def _resolve_sales_path() -> Path:
    """
    Resolve the main rolling sales file.
    """
    csv_path = Path(ROLLING_SALES_CSV)
    p_parq = csv_path.with_suffix(".parquet")
    p_feat = csv_path.with_suffix(".feather")
    if p_parq.exists():
        return p_parq
    if p_feat.exists():
        return p_feat
    return csv_path


@st.cache_data(show_spinner=False)
def load_raw_sales() -> pd.DataFrame:
    """
    Load raw sales data from CSV / parquet / feather and normalize columns.
    """
    p = _resolve_sales_path()
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)  # this requires pyarrow OR fastparquet (on your own machine)
    elif p.suffix == ".feather":
        df = pd.read_feather(p)
    else:
        df = pd.read_csv(p)

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    required = {"ADDRESS", "ZIP CODE", "SALE PRICE"}
    if not required.issubset(set(df.columns)):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Input file missing columns: {missing}")

    # Basic cleaning
    df["SALE PRICE"] = pd.to_numeric(df["SALE PRICE"], errors="coerce")
    df = df[df["SALE PRICE"] >= 100000].dropna(
        subset=["ADDRESS", "ZIP CODE", "SALE PRICE"]
    )

    df["ZIP CODE"] = (
        df["ZIP CODE"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
    )

    for col in ["GROSS SQUARE FEET", "YEAR BUILT", "TOTAL UNITS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    if "BUILDING CLASS CATEGORY" not in df.columns:
        df["BUILDING CLASS CATEGORY"] = ""

    if "SALE DATE" in df.columns:
        df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors="coerce")
    else:
        df["SALE DATE"] = pd.NaT

    return df


def _est_year_built(v):
    return 1960 if pd.isna(v) or v <= 0 else int(v)


def _est_sqft(v):
    return 900 if pd.isna(v) or v <= 0 else int(v)


def _est_beds(row):
    cat = str(row.get("BUILDING CLASS CATEGORY", "")).upper()
    units = row.get("TOTAL UNITS")
    if "ONE FAMILY" in cat:
        return 3
    if "TWO FAMILY" in cat:
        return 4
    if "THREE FAMILY" in cat:
        return 5
    if "COND" in cat or "COOP" in cat:
        return 2
    if pd.notna(units):
        if units <= 1:
            return 2
        if units == 2:
            return 3
        if units <= 4:
            return 4
        return 5
    return 2


def _est_baths(beds):
    return 2.0 if beds >= 4 else (1.5 if beds >= 2 else 1.0)


APT_PAT = re.compile(r"\b(apt|unit|#)\s*[\w\-]+", re.IGNORECASE)


def _norm_addr(s: str) -> str:
    x = str(s)
    x = APT_PAT.sub("", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x.title()


@st.cache_data(show_spinner=False)
def build_listings_base() -> pd.DataFrame:
    """
    Build normalized listing table (no geometry yet).
    """
    df = load_raw_sales().copy()
    ranks = pd.Series(range(1, len(df) + 1), index=df.index)
    dom = 10 + (ranks % 81)

    lst = pd.DataFrame(
        {
            "postcode": df["ZIP CODE"],
            "address": df["ADDRESS"].astype(str).str.title(),
            "year_built": df["YEAR BUILT"].apply(_est_year_built),
            "price": df["SALE PRICE"].round(0).astype(int),
            "square_feet": df["GROSS SQUARE FEET"].apply(_est_sqft),
            "neighborhood": df.get("NEIGHBORHOOD", pd.NA),
            "building_class_category": df.get("BUILDING CLASS CATEGORY", pd.NA),
            "sale_date": df.get("SALE DATE", pd.NaT),
            "days_on_market": dom,
        }
    )
    lst["beds"] = df.apply(_est_beds, axis=1)
    lst["baths"] = lst["beds"].apply(_est_baths)
    lst["full_address"] = (lst["address"] + ", " + lst["postcode"]).apply(_norm_addr)
    return lst


def _load_cache_map() -> Dict[str, Tuple[float, float]]:
    """
    Load cached lat/lon for full_address.
    """
    m: Dict[str, Tuple[float, float]] = {}

    base = Path(GEOCODE_CACHE)  # config path without extension
    p_parq = base.with_suffix(".parquet")
    p_csv = base.with_suffix(".csv")

    # 1 Try parquet
    if p_parq.exists():
        try:
            c = pd.read_parquet(p_parq)
            m.update(
                {
                    fa: (float(lat), float(lon))
                    for fa, lat, lon in zip(c["full_address"], c["lat"], c["lon"])
                }
            )
        except Exception:
            # If parquet engine is missing on another machine, silently fall back to CSV
            pass

    # 2 Fallback to CSV
    if not m and p_csv.exists():
        s = pd.read_csv(p_csv)
        if {"full_address", "lat", "lon"}.issubset(set(s.columns)):
            s["full_address"] = s["full_address"].astype(str).apply(_norm_addr)
            m.update(
                {
                    fa: (float(lat), float(lon))
                    for fa, lat, lon in zip(s["full_address"], s["lat"], s["lon"])
                }
            )

    # 3 GEO_SEED_CSV as extra fallback and merge
    if Path(GEO_SEED_CSV).exists():
        s = pd.read_csv(GEO_SEED_CSV)
        if {"full_address", "lat", "lon"}.issubset(set(s.columns)):
            s["full_address"] = s["full_address"].astype(str).apply(_norm_addr)
            m.update(
                {
                    fa: (float(lat), float(lon))
                    for fa, lat, lon in zip(s["full_address"], s["lat"], s["lon"])
                }
            )

    return m


@st.cache_data(show_spinner=False)
def attach_geometry(use_address_geocoding: bool = True) -> pd.DataFrame:
    """
    Build listing table with lat/lon attached.
    Returns a plain pandas.DataFrame (no GeoDataFrame).
    """
    lst = build_listings_base()
    raw = load_raw_sales()

    # If raw has LAT or LON columns, prefer them
    if {"LAT", "LON"}.issubset(set(raw.columns)):
        raw_key = (
            raw["ADDRESS"].astype(str).str.title()
            + ", "
            + raw["ZIP CODE"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(5)
        ).apply(_norm_addr)
        key_to_ll = dict(
            zip(
                raw_key,
                zip(
                    pd.to_numeric(raw["LAT"], errors="coerce"),
                    pd.to_numeric(raw["LON"], errors="coerce"),
                ),
            )
        )
        latlon = [key_to_ll.get(a, (np.nan, np.nan)) for a in lst["full_address"]]
        lst["lat"] = [p[0] for p in latlon]
        lst["lon"] = [p[1] for p in latlon]
    else:
        # Fallback using pre-geocoded cache files
        cache_map = _load_cache_map()
        latlon = [cache_map.get(a, (np.nan, np.nan)) for a in lst["full_address"]]
        lst["lat"] = [p[0] for p in latlon]
        lst["lon"] = [p[1] for p in latlon]

    # Sanity check
    missing = int(lst["lat"].isna().sum() + lst["lon"].isna().sum()) // 2
    if FAST_START_STRICT and missing > 0:
        raise RuntimeError(
            f"Found {missing} listings without coordinates. "
            f"Run tools/pregeocode_all.py to build data/geocode_cache.(parquet/csv), "
            f"or add LAT/LON into the CSV. (Instant-start does not call live geocoding.)"
        )

    lst = lst.dropna(subset=["lat", "lon"])
    return lst


def get_filter_options(df: pd.DataFrame):
    return {"postcodes": sorted(df["postcode"].dropna().unique().tolist())}
