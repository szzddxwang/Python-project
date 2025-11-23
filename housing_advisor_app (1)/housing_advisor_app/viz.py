import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import streamlit as st

# Prefer st_folium; fallback to folium_static if unavailable
try:
    from streamlit_folium import st_folium, folium_static
except Exception:  # pragma: no cover, fallback only
    st_folium = None
    folium_static = None


def render_map(df: pd.DataFrame, highlight_address=None, max_points: int = 3000):
    """
    Render a Folium map:
    - Plot markers only from the given df (df already filtered by ZIP and other conditions)
    - max_points: upper bound to avoid loading too many points at once (lag or nothing shown)
    - highlight_address: additionally highlight one listing on the map with a CircleMarker
    """
    if df.empty:
        st.info("No data to show on map.")
        return None

    # Limit maximum number of points to avoid heavy front end rendering
    if len(df) > max_points:
        df_show = df.head(max_points).copy()
        st.caption(
            f"Showing first {len(df_show)} points out of {len(df)} for performance. "
            f"Use the slider to increase if needed."
        )
    else:
        df_show = df

    center = [df_show["lat"].mean(), df_show["lon"].mean()]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    cluster = MarkerCluster(name="Listings").add_to(m)
    for _, r in df_show.iterrows():
        html = (
            f"{r['full_address']}<br>"
            f"${r['price']:,.0f}<br>"
            f"{r['beds']} bd / {r['baths']} ba, {int(r['square_feet'])} sqft"
        )
        folium.Marker(
            location=[r["lat"], r["lon"]],
            tooltip=r["full_address"],
            popup=folium.Popup(html, max_width=350),
        ).add_to(cluster)

    # If highlight_address is provided, draw a circle to highlight it
    if highlight_address:
        row = df_show[df_show["full_address"] == highlight_address].head(1)
        if not row.empty:
            folium.CircleMarker(
                location=[row.iloc[0]["lat"], row.iloc[0]["lon"]],
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
            st.error("streamlit-folium is not installed correctly.")
            return None
    except Exception as e:  # pragma: no cover - runtime fallback
        st.warning(f"Primary renderer failed: {e}. Falling back to folium_static.")
        if folium_static is not None:
            folium_static(m, width=960, height=620)
            return None
        st.error("Cannot render map. Please ensure 'streamlit-folium' is installed.")
        return None


def find_listing_from_click(map_data, df: pd.DataFrame):
    """
    Given a clicked location on the map, find the nearest listing in df.
    Use NumPy vectorized distance computation to significantly speed up selection when many points exist.
    """
    if not map_data or "last_object_clicked" not in map_data or map_data["last_object_clicked"] is None:
        return None

    lat = map_data["last_object_clicked"]["lat"]
    lng = map_data["last_object_clicked"]["lng"]

    if df.empty:
        return None

    # Use vectorized distance computation to speed up nearest-point search
    coords = df[["lat", "lon"]].to_numpy()
    dists = np.abs(coords[:, 0] - lat) + np.abs(coords[:, 1] - lng)
    idx = int(dists.argmin())

    row = df.iloc[idx]
    return row.to_dict()
