"""Streamlit Real-Time Shelf Monitor & Presentation Dashboard.

Run with:
    streamlit run scripts/dashboard.py
"""
import json
from pathlib import Path
import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Sensei Produce Monitor",
    page_icon="🥑",
    layout="wide",
)

st.title("🛒 Sensei Shelf Intelligence — Continente Bom Dia")
st.markdown("Autonomous produce monitoring, compliance verification, and real-time stock alerting.")

PRED_PATH = Path("output/predictions.json")
PLANO_PATH = Path("planogram.json")

if not PRED_PATH.exists():
    st.warning("No predictions found in `output/predictions.json`. Run pipeline first!")
    st.stop()

with open(PRED_PATH) as f:
    predictions = json.load(f)

planogram_data = {}
if PLANO_PATH.exists():
    with open(PLANO_PATH) as f:
        planogram_data = json.load(f)

# Global metrics calculation
total_areas = sum(len(areas) for cam in predictions.values() for areas in cam.values())
empty_count = sum(1 for cam in predictions.values() for areas in cam.values() for a in areas.values() if a.get("stock_level") == "empty")
compliant_count = sum(1 for cam in predictions.values() for areas in cam.values() for a in areas.values() if a.get("planogram_compliant"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cameras Monitored", len(predictions))
col2.metric("Total Shelf Slots", total_areas)
col3.metric("Planogram Compliance", f"{(compliant_count/total_areas*100):.1f}%" if total_areas else "0%")
col4.metric("Immediate Restock Needed", empty_count)

st.divider()

# Camera selector
selected_cam = st.sidebar.selectbox("Select Camera", sorted(list(predictions.keys())))
images_for_cam = predictions.get(selected_cam, {})
selected_img = st.sidebar.selectbox("Select Timestamp", sorted(list(images_for_cam.keys())))

# Display Area Breakdown
st.subheader(f"Camera {selected_cam} — Image: `{selected_img}`")

areas_pred = images_for_cam.get(selected_img, {})

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 📋 Area Prediction Status")
    table_data = []
    for area_id, data in areas_pred.items():
        table_data.append({
            "Area ID": area_id,
            "Compliant": "✅ Yes" if data.get("planogram_compliant") else "❌ No",
            "Stock Level": data.get("stock_level", "").upper(),
            "Fill Value": f"{data.get('stock_value', 0.0) * 100:.0f}%",
        })
    st.dataframe(table_data, use_container_width=True)

with c2:
    st.markdown("### 🚨 Urgent Interventions")
    urgent = [a for a, d in areas_pred.items() if d.get("stock_level") == "empty" or not d.get("planogram_compliant")]
    if urgent:
        for u in urgent:
            d = areas_pred[u]
            reason = []
            if d.get("stock_level") == "empty":
                reason.append("CRITICAL OUT-OF-STOCK")
            if not d.get("planogram_compliant"):
                reason.append("PRODUCT MISMATCH")
            st.error(f"**Slot {u}**: {' & '.join(reason)}")
    else:
        st.success("All slots in this view are healthy and compliant!")
