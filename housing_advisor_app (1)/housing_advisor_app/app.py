import streamlit as st
import pandas as pd

from config import APP_TITLE, ensure_data_exists
from data_io import attach_geometry
from models_trend import compute_trend_score
from models_ratio import price_to_rent_ratio, ratio_score, affordability_score
from decision_engine import DecisionEngine, DecisionInputs
from viz import render_map, find_listing_from_click

# ---- App setup ----
ensure_data_exists()
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("Manual ZIP search + advanced filters. LAT/LON loaded from CSV/cache (no live geocoding during search).")

engine = DecisionEngine()

# ---- Load data with geometry ----
try:
    gdf = attach_geometry(use_address_geocoding=True)
except Exception as e:
    st.error(str(e))
    st.stop()

if gdf.empty:
    st.error("No listings with coordinates. Prepare LAT/LON or run tools/pregeocode_all.py.")
    st.stop()

# Basic column checks (defensive)
required_cols = ["postcode", "full_address", "price", "beds", "baths", "square_feet", "days_on_market", "lat", "lon"]
missing = [c for c in required_cols if c not in gdf.columns]
if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

# ---- Sidebar: manual ZIP + 原有筛选条件 ----
st.sidebar.header("Search by ZIP (Postcode)")

zip_input = st.sidebar.text_input("ZIP / Postcode (required)", value="", max_chars=20)
search_button = st.sidebar.button("Search", use_container_width=True)

# 控制地图一次渲染的最大点数，避免过多点导致卡顿或不显示
max_points = st.sidebar.slider("Max points on map", 500, 15000, 5000, 500)

# 使用 session_state 存储当前生效的 ZIP，避免每次输入字符就重新筛选
if "active_zip" not in st.session_state:
    st.session_state["active_zip"] = None

if search_button:
    z = zip_input.strip()
    st.session_state["active_zip"] = z if z else None

active_zip = st.session_state.get("active_zip")

if not active_zip:
    st.info("Please enter（ZIP / Postcode），then click **Search** to start searching for properties。")
    st.stop()

# 先按 ZIP 过滤，再做其他条件
df_zip = gdf[gdf["postcode"].astype(str) == str(active_zip)]

if df_zip.empty:
    st.warning(f"Postcode = {active_zip} No listings found. Please try other postcodes.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Additional filters")

# 价格范围
price_min = int(df_zip["price"].min())
price_max = int(df_zip["price"].max())
price_range = st.sidebar.slider(
    "Price range ($)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=max(5000, (price_max - price_min) // 100 or 1000),
)

# 面积范围
if df_zip["square_feet"].notna().any():
    sqft_min = int(df_zip["square_feet"].min())
    sqft_max = int(df_zip["square_feet"].max())
else:
    sqft_min, sqft_max = 0, 0
sqft_range = st.sidebar.slider(
    "Square feet range",
    min_value=sqft_min,
    max_value=max(sqft_max, sqft_min + 1),
    value=(sqft_min, max(sqft_max, sqft_min + 1)),
    step=max(10, (sqft_max - sqft_min) // 100 or 10),
)

# 卧室/卫生间
beds_min = int(df_zip["beds"].min())
beds_max = int(df_zip["beds"].max())
baths_min = int(df_zip["baths"].min())
baths_max = int(df_zip["baths"].max())

min_beds = st.sidebar.number_input("Min beds", min_value=0, max_value=beds_max, value=beds_min, step=1)
min_baths = st.sidebar.number_input("Min baths", min_value=0, max_value=baths_max, value=baths_min, step=1)

# 地址关键词
keyword = st.sidebar.text_input("Address keyword contains", value="").strip()

# ---- 应用所有过滤条件 ----
df = df_zip.copy()
df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
df = df[(df["square_feet"] >= sqft_range[0]) & (df["square_feet"] <= sqft_range[1])]
df = df[df["beds"] >= min_beds]
df = df[df["baths"] >= min_baths]
if keyword:
    df = df[df["full_address"].str.contains(keyword, case=False, na=False)]

if df.empty:
    st.warning("No listings are available under the current postal code and filter conditions. Please broaden the price/bedroom/storage area or change the postal code.。")
    st.stop()

st.write(
    f"current postal code **{active_zip}** has following eligible properties：**{len(df)}**  "
    f"(This postal code has {len(df_zip)} houses，The number of entire database with coordinates {len(gdf)} )。"
)

# ---- Map + Table layout ----
col_map, col_table = st.columns([2, 1.7])

with col_map:
    # 默认高亮第一套房源
    hl = df.iloc[0]["full_address"] if not df.empty else None
    map_data = render_map(
        df,
        selected_postcode=active_zip,
        highlight_address=hl,
        max_points=max_points,
    )
    clicked_listing = find_listing_from_click(map_data, df) if map_data and not df.empty else None

with col_table:
    st.subheader("Listings")
    if df.empty:
        st.info("No listings under current filters.")
    else:
        show_cols = ["full_address", "postcode", "price", "beds", "baths", "square_feet", "days_on_market"]
        display = df[show_cols].copy()
        display["price"] = display["price"].map(lambda x: f"${x:,.0f}")
        st.dataframe(display, use_container_width=True, height=430)

    if clicked_listing is not None:
        st.markdown("**Selected from map:**")
        details = (
            f"{clicked_listing['full_address']} | "
            f"${clicked_listing['price']:,.0f} | "
            f"{clicked_listing['beds']} bd / {clicked_listing['baths']} ba, "
            f"{int(clicked_listing['square_feet'])} sqft, "
            f"{int(clicked_listing['days_on_market'])} days on market"
        )
        st.code(details, language="text")

# ---- Advisor ----
st.markdown('---')
st.subheader('Buy-Timing Advisor (3-factor)')

# 优先使用地图点击的房源；没有点击时使用当前结果中的第一套
target_listing = clicked_listing if 'clicked_listing' in locals() and clicked_listing is not None else (df.iloc[0] if not df.empty else None)

c1, c2, c3 = st.columns(3)
with c1:
    user_income = st.number_input("Your annual income ($)", min_value=0, value=100000, step=5000)
with c2:
    est_monthly_rent = st.number_input("Comparable monthly rent ($)", min_value=0, value=2500, step=100)
with c3:
    run_button = st.button("Run Buy/Watch/Avoid Analysis", use_container_width=True)

if target_listing is not None and run_button:
    same_zip = gdf[gdf["postcode"] == target_listing["postcode"]]
    ts = same_zip["price"].sort_values()
    t_score = compute_trend_score(ts)

    ptr = price_to_rent_ratio(price=target_listing["price"], monthly_rent=est_monthly_rent or None)
    r_score = ratio_score(ptr)

    a_score = affordability_score(price=target_listing["price"], income=user_income or None)

    result = engine.decide(DecisionInputs(t_score, r_score, a_score))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Trend score", f"{t_score:.2f}")
    with m2:
        st.metric("P/R ratio", f"{ptr:.1f}" if (ptr is not None) else "N/A")
    with m3:
        st.metric("Affordability score", f"{a_score:.2f}")

    st.markdown(f"### Recommendation: **{result.label}**  (composite score {result.score:.3f})")
    st.write(result.explanation)

    st.info(
        f"Target listing: {target_listing['full_address']}  |  "
        f"${target_listing['price']:,.0f}, "
        f"{target_listing['beds']} bd / {target_listing['baths']} ba, "
        f"{int(target_listing['square_feet'])} sqft."
    )

elif run_button and target_listing is None:
    st.warning("No target listing available under current filters. Click a point or relax filters.")
