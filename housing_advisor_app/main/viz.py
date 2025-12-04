import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import streamlit as st

try:
    from streamlit_folium import st_folium, folium_static
except Exception:  # pragma: no cover
    st_folium = None
    folium_static = None


def _rank_div_icon(rank: int) -> folium.DivIcon:
    # A clean circular badge with a rank
    html = f"""
    <div style="
        width:30px;height:30px;
        border-radius:9999px;
        background:#ff6b00;
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:800;
        border:2px solid rgba(255,255,255,0.95);
        box-shadow:0 2px 8px rgba(0,0,0,0.25);
        font-size:14px;
        ">
        {rank}
    </div>
    """
    return folium.DivIcon(html=html)


def render_map(
    df: pd.DataFrame,
    highlight_address: str | None = None,
    top_ranked_addresses: list[str] | None = None,  # ordered list
    max_points: int = 3000,
):

    if df is None or df.empty:
        st.info("No data to show on map.")
        return None

    # Keep Top-N rows visible even if we truncate df for performance
    top_set = set(top_ranked_addresses or [])
    df_top = df[df["full_address"].astype(str).isin(top_set)].copy() if top_set else df.head(0).copy()

    # Limit maximum points for performance
    if len(df) > max_points:
        df_show = df.head(max_points).copy()
        # Make sure top-ranked rows are included
        if not df_top.empty:
            df_show = pd.concat([df_show, df_top], ignore_index=True).drop_duplicates(subset=["full_address"])
        st.caption(
            f"Showing {len(df_show)} points (limited for performance). Increase 'Max points on map' if needed."
        )
    else:
        df_show = df.copy()

    center = [float(df_show["lat"].mean()), float(df_show["lon"].mean())]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    cluster = MarkerCluster(name="Listings").add_to(m)
    has_ml = "ml_match_score" in df_show.columns

    # 1 Add clustered markers
    for _, r in df_show.iterrows():
        addr = str(r["full_address"])
        if addr in top_set:
            continue  # do not duplicate; will be rendered as numbered marker below

        html_parts = [
            f"<b>{addr}</b>",
            f"${float(r['price']):,.0f}",
            f"{int(r['beds'])} bd / {int(r['baths'])} ba, {int(r['square_feet'])} sqft",
        ]
        if has_ml and pd.notna(r.get("ml_match_score", np.nan)):
            html_parts.append(f"ML match score: {float(r['ml_match_score']):.3f}")
        html = "<br>".join(html_parts)

        folium.Marker(
            location=[float(r["lat"]), float(r["lon"])],
            tooltip=addr,
            popup=folium.Popup(html, max_width=380),
        ).add_to(cluster)

    # 2 Overlay Top-N numbered markers, unclustered and always visible
    if top_ranked_addresses:
        for i, addr in enumerate(top_ranked_addresses, start=1):
            row = df_show[df_show["full_address"].astype(str) == str(addr)].head(1)
            if row.empty:
                continue
            r = row.iloc[0]

            html_parts = [
                f"<span style='color:#ff6b00; font-weight:800;'>Top #{i}</span>",
                f"<b>{str(r['full_address'])}</b>",
                f"${float(r['price']):,.0f}",
                f"{int(r['beds'])} bd / {int(r['baths'])} ba, {int(r['square_feet'])} sqft",
            ]
            if has_ml and pd.notna(r.get("ml_match_score", np.nan)):
                html_parts.append(f"ML match score: {float(r['ml_match_score']):.3f}")
            html = "<br>".join(html_parts)

            folium.Marker(
                location=[float(r["lat"]), float(r["lon"])],
                tooltip=f"Top #{i}: {str(r['full_address'])}",
                popup=folium.Popup(html, max_width=420),
                icon=_rank_div_icon(i),
                z_index_offset=1000 + (10 - i),
            ).add_to(m)

    # 3 Highlight circle marker
    if highlight_address:
        row = df_show[df_show["full_address"] == highlight_address].head(1)
        if not row.empty:
            folium.CircleMarker(
                location=[float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])],
                radius=8,
                weight=3,
                color="#ff6b00",
                fill=True,
                fill_opacity=0.6,
            ).add_to(m)

    # Render
    try:
        if st_folium is not None:
            return st_folium(m, width=960, height=620)
        elif folium_static is not None:
            folium_static(m, width=960, height=620)
            return None
        else:
            st.warning("streamlit-folium is not available; cannot render map interactively.")
            return None
    except Exception as e:  # pragma: no cover
        st.warning(f"Map render failed: {e}")
        return None


def find_listing_from_click(map_data, df: pd.DataFrame):
    if map_data is None or df is None or df.empty:
        return None
    if not isinstance(map_data, dict):
        return None
    if "last_object_clicked" not in map_data or map_data["last_object_clicked"] is None:
        return None

    lat = map_data["last_object_clicked"].get("lat")
    lng = map_data["last_object_clicked"].get("lng")
    if lat is None or lng is None:
        return None

    coords = df[["lat", "lon"]].astype(float).to_numpy()
    dists = np.abs(coords[:, 0] - float(lat)) + np.abs(coords[:, 1] - float(lng))
    idx = int(dists.argmin())
    return df.iloc[idx].to_dict()
