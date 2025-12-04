import streamlit as st
import numpy as np
import hashlib
from config import APP_TITLE, ensure_data_exists
from data_io import attach_geometry
from models_trend import compute_trend_score
from models_ratio import price_to_rent_ratio, ratio_score, affordability_score
from decision_engine import DecisionEngine, DecisionInputs
from viz import render_map, find_listing_from_click
from Neuralnetwork import get_ranker, save_ranker, extract_features


def _k(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _signature(active_zip: str, price_range, sqft_range, min_beds, min_baths, keyword: str):
    return (
        str(active_zip),
        int(price_range[0]), int(price_range[1]),
        int(sqft_range[0]), int(sqft_range[1]),
        int(min_beds), int(min_baths),
        str(keyword or "").strip().lower(),
    )


def _init_state():
    st.session_state.setdefault("active_zip", None)
    st.session_state.setdefault("clicked_address", None)

    # Advisor inputs + last output
    st.session_state.setdefault("advisor_income", 100000)
    st.session_state.setdefault("advisor_rent", 2500)
    st.session_state.setdefault("advisor_last", None)

    # Personalized finder inputs + last output
    st.session_state.setdefault("ml_target_price", 0)
    st.session_state.setdefault("ml_target_rent", 2500)
    st.session_state.setdefault("ml_income", 100000)
    st.session_state.setdefault("ml_tax_rate", 20.00)
    st.session_state.setdefault("ml_last", None)

    # Table sort
    st.session_state.setdefault("sort_by_ml", True)

    # Teach-the-model vote lock
    st.session_state.setdefault("nn_feedback", {})


#  App setup
ensure_data_exists()
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

_init_state()
engine = DecisionEngine()

try:
    gdf = attach_geometry(use_address_geocoding=True)
except Exception as e:
    st.error(str(e))
    st.stop()

if gdf.empty:
    st.error("No listings with coordinates. Prepare LAT/LON or run tools/pregeocode_all.py.")
    st.stop()

required_cols = ["postcode", "full_address", "price", "beds", "baths", "square_feet", "days_on_market", "lat", "lon"]
missing = [c for c in required_cols if c not in gdf.columns]
if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

#Sidebar search + filters
st.sidebar.header("Search by ZIP (Postcode)")
zip_input = st.sidebar.text_input("ZIP / Postcode (required)", value="", max_chars=20, key="zip_input")
search_button = st.sidebar.button("Search", use_container_width=True, key="zip_search_btn")
max_points = st.sidebar.slider("Max points on map", 500, 15000, 5000, 500, key="max_points")

if search_button:
    z = (zip_input or "").strip()
    st.session_state["active_zip"] = z if z else None
    st.session_state["clicked_address"] = None
    # NOTE: we intentionally do NOT clear inputs/results; only map selection resets

active_zip = st.session_state.get("active_zip")
if not active_zip:
    st.info("Please enter ZIP / Postcode, then click **Search**.")
    st.stop()

df_zip = gdf[gdf["postcode"].astype(str) == str(active_zip)]
if df_zip.empty:
    st.warning(f"Postcode = {active_zip} : no listings found. Try another postcode.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Additional filters")

price_min = int(df_zip["price"].min())
price_max = int(df_zip["price"].max())
price_step = max(5000, (price_max - price_min) // 100 or 1000)
price_range = st.sidebar.slider(
    "Price range ($)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=price_step,
    key="filter_price_range",
)

if df_zip["square_feet"].notna().any():
    sqft_min = int(df_zip["square_feet"].min())
    sqft_max = int(df_zip["square_feet"].max())
else:
    sqft_min, sqft_max = 0, 1

sqft_step = max(10, (sqft_max - sqft_min) // 100 or 10)
sqft_range = st.sidebar.slider(
    "Square feet range",
    min_value=sqft_min,
    max_value=max(sqft_max, sqft_min + 1),
    value=(sqft_min, max(sqft_max, sqft_min + 1)),
    step=sqft_step,
    key="filter_sqft_range",
)

beds_min = int(df_zip["beds"].min())
beds_max = int(df_zip["beds"].max())
baths_min = int(df_zip["baths"].min())
baths_max = int(df_zip["baths"].max())
min_beds = st.sidebar.number_input("Min beds", min_value=0, max_value=beds_max, value=beds_min, step=1, key="filter_min_beds")
min_baths = st.sidebar.number_input("Min baths", min_value=0, max_value=baths_max, value=baths_min, step=1, key="filter_min_baths")

keyword = st.sidebar.text_input("Address keyword contains", value="", key="filter_keyword").strip()

df = df_zip.copy()
df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
df = df[(df["square_feet"] >= sqft_range[0]) & (df["square_feet"] <= sqft_range[1])]
df = df[df["beds"] >= min_beds]
df = df[df["baths"] >= min_baths]
if keyword:
    df = df[df["full_address"].str.contains(keyword, case=False, na=False)]

if df.empty:
    st.warning("No listings under current filters. Broaden the filters or change ZIP.")
    st.stop()

sig = _signature(active_zip, price_range, sqft_range, min_beds, min_baths, keyword)

# Default target price once per session
if st.session_state.get("ml_target_price", 0) in (0, None):
    st.session_state["ml_target_price"] = int(df["price"].median())

st.write(
    f"Postcode **{active_zip}** eligible properties: **{len(df)}**  "
    f"(Total in this ZIP: {len(df_zip)}, total with coordinates: {len(gdf)})"
)

# If last ML run matches signature, use scored df for display
df_use = df
ml_last = st.session_state.get("ml_last")
if isinstance(ml_last, dict) and ml_last.get("signature") == sig and ml_last.get("scored_df") is not None:
    df_use = ml_last["scored_df"]

top5_addrs = []
if isinstance(ml_last, dict) and ml_last.get("signature") == sig:
    top5_addrs = ml_last.get("top5_addresses", []) or []

# Map + listings table
col_map, col_table = st.columns([2, 1.7])

with col_map:
    best_addr = ml_last.get("best_address") if isinstance(ml_last, dict) else None
    clicked_addr = st.session_state.get("clicked_address")

    if best_addr and best_addr in df_use["full_address"].values:
        hl = best_addr
    elif clicked_addr and clicked_addr in df_use["full_address"].values:
        hl = clicked_addr
    else:
        hl = df_use.iloc[0]["full_address"]

    map_data = render_map(
        df_use,
        highlight_address=hl,
        top_ranked_addresses=top5_addrs,  # show 1–5 on map
        max_points=int(st.session_state["max_points"]),
    )

    clicked_listing = find_listing_from_click(map_data, df_use) if map_data and not df_use.empty else None
    if clicked_listing is not None and "full_address" in clicked_listing:
        st.session_state["clicked_address"] = clicked_listing["full_address"]

with col_table:
    st.subheader("Listings")

    has_ml = "ml_match_score" in df_use.columns

    df_view = df_use.copy()
    if has_ml:
        df_view = df_view.sort_values("ml_match_score", ascending=False, kind="mergesort")

    show_cols = ["full_address", "postcode", "price", "beds", "baths", "square_feet", "days_on_market"]
    if has_ml:
        show_cols = ["ml_match_score"] + show_cols

    display = df_view[show_cols].copy()
    if has_ml:
        display["ml_match_score"] = display["ml_match_score"].map(lambda x: f"{float(x):.3f}")
    display["price"] = display["price"].map(lambda x: f"${float(x):,.0f}")

    st.dataframe(display, use_container_width=True, height=430)

    ca = st.session_state.get("clicked_address")
    if ca:
        row = df_use[df_use["full_address"].astype(str) == str(ca)].head(1)
        if not row.empty:
            r = row.iloc[0]
            st.markdown("**Selected from map:**")
            st.code(
                f"{r['full_address']} | ${float(r['price']):,.0f} | "
                f"{int(r['beds'])} bd / {int(r['baths'])} ba, "
                f"{int(r['square_feet'])} sqft, {int(r['days_on_market'])} days on market",
                language="text",
            )

#  Advisor
st.markdown("---")
st.subheader("Buy-Timing Advisor")

c1, c2, c3 = st.columns(3)
with c1:
    st.number_input("Your annual income ($)", min_value=0, step=5000, key="advisor_income")
with c2:
    st.number_input("Comparable monthly rent ($)", min_value=0, step=100, key="advisor_rent")
with c3:
    run_button = st.button("Run Buy/Watch/Avoid Analysis", use_container_width=True, key="advisor_run_btn")

target_addr = st.session_state.get("clicked_address")
if target_addr and target_addr in df_use["full_address"].values:
    target_listing = df_use[df_use["full_address"] == target_addr].iloc[0].to_dict()
else:
    target_listing = df_use.iloc[0].to_dict() if not df_use.empty else None

if run_button and target_listing is not None:
    same_zip = gdf[gdf["postcode"] == target_listing["postcode"]]
    ts = same_zip["price"].sort_values()
    t_score = float(compute_trend_score(ts))

    ptr = price_to_rent_ratio(price=target_listing["price"], monthly_rent=st.session_state["advisor_rent"] or None)
    a_score = float(affordability_score(price=target_listing["price"], income=st.session_state["advisor_income"] or None))
    result = engine.decide(DecisionInputs(t_score, float(ratio_score(ptr)), a_score))

    st.session_state["advisor_last"] = {
        "address": target_listing["full_address"],
        "price": float(target_listing["price"]),
        "beds": int(target_listing["beds"]),
        "baths": int(target_listing["baths"]),
        "sqft": int(target_listing["square_feet"]),
        "t_score": t_score,
        "ptr": None if ptr is None else float(ptr),
        "a_score": a_score,
        "label": result.label,
        "composite": float(result.score),
        "explanation": result.explanation,
    }

adv_last = st.session_state.get("advisor_last")
if isinstance(adv_last, dict):
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Trend score", f"{adv_last['t_score']:.2f}")
    with m2:
        st.metric("P/R ratio", f"{adv_last['ptr']:.1f}" if (adv_last["ptr"] is not None) else "N/A")
    with m3:
        st.metric("Affordability score", f"{adv_last['a_score']:.2f}")

    st.markdown(f"### Recommendation: **{adv_last['label']}**  (composite score {adv_last['composite']:.3f})")
    st.write(adv_last["explanation"])
    st.info(
        f"Target listing: {adv_last['address']} | "
        f"${adv_last['price']:,.0f}, {adv_last['beds']} bd / {adv_last['baths']} ba, {adv_last['sqft']} sqft."
    )

#  Neural network with Teach-the-model lock
st.markdown("---")
st.subheader("Personalized Property Finder")

mcol1, mcol2 = st.columns(2)
with mcol1:
    st.number_input("Target purchase price ($)", min_value=0, step=5000, key="ml_target_price")
    st.number_input("Target monthly rent you expect ($)", min_value=0, step=100, key="ml_target_rent")
with mcol2:
    st.number_input("Your annual income before tax ($)", min_value=0, step=5000, key="ml_income")
    # ✅ user directly inputs percent (not slider)
    st.number_input(
        "Estimated personal income tax rate (%)",
        min_value=0.00,
        max_value=50.00,
        step=0.10,
        format="%.2f",
        key="ml_tax_rate",
        help="Personal income tax (effective %). Used to estimate AFTER-tax annual income for affordability.",
    )

ml_button = st.button("Find my best-matched property", use_container_width=True, key="ml_run_btn")

if ml_button:
    ranker = get_ranker()

    X, feat_names = extract_features(
        df,
        desired_price=st.session_state["ml_target_price"],
        desired_rent=st.session_state["ml_target_rent"],
        annual_income_before_tax=st.session_state["ml_income"],
        personal_income_tax_rate_pct=st.session_state["ml_tax_rate"],
    )
    scores = ranker.predict(X)

    scored_df = df.copy()
    scored_df["ml_match_score"] = scores

    best_row = scored_df.iloc[int(np.argmax(scores))]
    top5 = (
        scored_df.sort_values("ml_match_score", ascending=False)
        .head(5)["full_address"]
        .astype(str)
        .tolist()
    )

    feat_map = {str(addr): X[i, :].copy() for i, addr in enumerate(scored_df["full_address"].astype(str).tolist())}

    st.session_state["ml_last"] = {
        "signature": sig,
        "best_address": str(best_row["full_address"]),
        "best_row": best_row.to_dict(),
        "scored_df": scored_df,
        "top5_addresses": top5,
        "feat_map": feat_map,
        "feature_names": feat_names,
    }

ml_last = st.session_state.get("ml_last")
if isinstance(ml_last, dict) and ml_last.get("best_row") is not None:
    if ml_last.get("signature") != sig:
        st.warning("Result shown is from previous ZIP/filters. Press the button again to recompute.")

    br = ml_last["best_row"]
    st.success(
        f"Best match: {br['full_address']} "
        f"(${float(br['price']):,.0f}, {int(br['beds'])} bd / {int(br['baths'])} ba, {int(br['square_feet'])} sqft)"
    )

    scored_df = ml_last["scored_df"]
    top5_addrs = ml_last.get("top5_addresses", []) or []
    feat_map = ml_last.get("feat_map", {}) or {}

    if scored_df is not None and not scored_df.empty:
        top_df = scored_df.sort_values("ml_match_score", ascending=False).head(10).copy()
        top_df["price_fmt"] = top_df["price"].map(lambda x: f"${float(x):,.0f}")
        top_df["match_fmt"] = top_df["ml_match_score"].map(lambda x: f"{float(x):.3f}")
        st.markdown("**Top results (table):**")
        st.dataframe(
            top_df[["match_fmt", "full_address", "postcode", "price_fmt", "beds", "baths", "square_feet", "days_on_market"]],
            use_container_width=True,
            height=300,
        )

    #  Teach the model
    st.markdown("### Teach the model")
    st.caption("👍 / 👎: you can vote ONCE per property under current ZIP/filters (no repeat).")

    ranker = get_ranker()
    sig_id = _k(repr(sig))
    fb_map = st.session_state["nn_feedback"].setdefault(sig_id, {})  # address -> 1/-1

    if top5_addrs and scored_df is not None and not scored_df.empty:
        for rank, addr in enumerate(top5_addrs[:5], start=1):
            addr_str = str(addr)
            row = scored_df[scored_df["full_address"].astype(str) == addr_str].head(1)
            if row.empty:
                continue

            r = row.iloc[0]
            voted = fb_map.get(addr_str)  # None / 1 / -1
            voted_text = "✅ Voted 👍" if voted == 1 else ("✅ Voted 👎" if voted == -1 else "")

            left, mid, right = st.columns([6, 1.2, 1.2])
            with left:
                st.write(
                    f"**Top #{rank}** — {addr_str}  |  ${float(r['price']):,.0f}  |  score {float(r['ml_match_score']):.3f}  {voted_text}"
                )

            x = feat_map.get(addr_str)
            addr_key = _k(addr_str)

            # If already voted, then disable BOTH buttons
            disabled = voted is not None

            with mid:
                like_clicked = st.button("👍", key=f"like_{sig_id}_{addr_key}", disabled=disabled)
            with right:
                dislike_clicked = st.button("👎", key=f"dislike_{sig_id}_{addr_key}", disabled=disabled)

            if not disabled and x is not None:
                if like_clicked:
                    ranker.update_one(x, 1)
                    save_ranker(ranker)      # disk persistence
                    fb_map[addr_str] = 1     # lock this property
                    st.toast("Saved 👍", icon="✅")
                    st.rerun()

                elif dislike_clicked:
                    ranker.update_one(x, 0)
                    save_ranker(ranker)
                    fb_map[addr_str] = -1
                    st.toast("Saved 👎", icon="✅")
                    st.rerun()
