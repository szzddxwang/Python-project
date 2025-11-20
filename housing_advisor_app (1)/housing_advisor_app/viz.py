import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import streamlit as st

# 优先用 st_folium；不可用时退回 folium_static
try:
    from streamlit_folium import st_folium, folium_static
except Exception:  # pragma: no cover - fallback only
    st_folium = None
    folium_static = None


def render_map(df: pd.DataFrame, selected_postcode=None, highlight_address=None, max_points: int = 3000):
    """
    渲染 Folium 地图：
    - 只根据传入的 df 画点（df 已经过滤了 ZIP 和其他条件）
    - max_points：为避免一次性加载过多点（卡或不显示），做一个上限
    - highlight_address：在地图上额外用 CircleMarker 高亮一套房子
    """
    if df.empty:
        st.info("No data to show on map.")
        return None

    # 限制最大点数，避免前端一次渲染过重
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

    # 如果提供 highlight_address，则高亮画一个圆
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

    # 渲染（首选 st_folium，失败则 folium_static）
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
    根据地图点击位置，在 df 中找到最近的一套房源。
    使用 numpy 向量化计算距离，显著提升在大量点时的选中速度。
    """
    if not map_data or "last_object_clicked" not in map_data or map_data["last_object_clicked"] is None:
        return None

    lat = map_data["last_object_clicked"]["lat"]
    lng = map_data["last_object_clicked"]["lng"]

    if df.empty:
        return None

    # 使用向量化距离计算，加快最近点查找
    coords = df[["lat", "lon"]].to_numpy()
    dists = np.abs(coords[:, 0] - lat) + np.abs(coords[:, 1] - lng)
    idx = int(dists.argmin())

    row = df.iloc[idx]
    return row.to_dict()
