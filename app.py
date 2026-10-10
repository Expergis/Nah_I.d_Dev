"""Streamlit Dashboard - Real-time Produce Shelf Monitor & Gondola Compliance

Optimized for store staff decision-making and pitch presentation.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Setup page config
st.set_page_config(
    page_title="Sensei Produce Monitor | Continente S. Romão",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
PLANOGRAM_PATH = PROJECT_ROOT / "planogram.json"
PREDICTIONS_PATH = PROJECT_ROOT / "output" / "predictions.json"
DEMO_PATH = PROJECT_ROOT / "demo" / "demo_predictions.json"
DATA_DIR = PROJECT_ROOT / "data"

# Custom styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #28a745;
    }
    .alert-card {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_planogram_data():
    if PLANOGRAM_PATH.exists():
        with open(PLANOGRAM_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_predictions(use_demo: bool = False):
    target = DEMO_PATH if use_demo else PREDICTIONS_PATH
    if not target.exists() and DEMO_PATH.exists():
        target = DEMO_PATH
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4b/Continente_Logo.svg", width=180)
st.sidebar.title("🍌 Shelf Monitor")
st.sidebar.caption("Continente Bom Dia S. Romão — Leiria")

demo_mode = st.sidebar.toggle("⚡ Demo Mode (Pre-cached)", value=True, help="Toggle pre-cached results for instant pitch execution")

planogram_raw = load_planogram_data()
predictions = load_predictions(use_demo=demo_mode)

cameras_available = list(predictions.keys())
if not cameras_available and planogram_raw:
    cameras_available = list(planogram_raw.get("Frutas e Legumes", {}).get("cameras", {}).keys())
if not cameras_available:
    cameras_available = ["6215"]

selected_camera = st.sidebar.selectbox("Select Gondola Camera", options=cameras_available, index=0)

st.sidebar.divider()
st.sidebar.markdown("### 🎯 Challenge Rules")
st.sidebar.markdown(
    """
    - **Identification**: +1 pt correct, −2 pts wrong
    - **Stock Level**: +1 pt correct, −2 pts below, −3 pts above
    - **Missing Area**: −5 pts penalty
    """
)

# --- Header & Top Metrics ---
st.title("🛒 Gondola Real-Time Audit Dashboard")
st.caption(f"Camera ID: **{selected_camera}** | Active Mode: **{'Pre-cached Demo' if demo_mode else 'Live Pipeline'}**")

# Collect areas for this camera
cam_preds = predictions.get(selected_camera, {})
first_image = list(cam_preds.keys())[0] if cam_preds else "1751052241.jpg"
image_areas = cam_preds.get(first_image, {})

total_areas = len(image_areas)
compliant_count = sum(1 for a in image_areas.values() if a.get("planogram_compliant", True))
empty_count = sum(1 for a in image_areas.values() if a.get("stock_value") == 0.0)
half_count = sum(1 for a in image_areas.values() if a.get("stock_value") == 0.5)

col1, col2, col3, col4 = st.columns(4)
with col1:
    compliance_rate = (compliant_count / total_areas * 100) if total_areas > 0 else 100.0
    st.metric("Planogram Compliance", f"{compliance_rate:.0f}%", delta=f"{compliant_count}/{total_areas} areas")
with col2:
    st.metric("Critical Restocks (Empty)", f"{empty_count}", delta="-Priority Urgent" if empty_count > 0 else "OK", delta_color="inverse")
with col3:
    st.metric("Half Stock Warnings", f"{half_count}", delta="Monitor" if half_count > 0 else "Optimal")
with col4:
    full_count = total_areas - empty_count - half_count
    st.metric("Optimal Stock (Full)", f"{full_count}", delta=f"{full_count}/{total_areas}")

st.divider()

# --- Main Visuals & Gondola Breakdown ---
v_col, d_col = st.columns([3, 2])

with v_col:
    st.subheader("📷 Camera Field of View & Quadrilateral Overlays")

    # Attempt to locate actual image
    sample_img_path = DATA_DIR / "validation" / selected_camera / first_image
    if not sample_img_path.exists():
        sample_img_path = DATA_DIR / "train" / selected_camera / first_image

    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Feed: {first_image} (Camera {selected_camera})", use_container_width=True)
    else:
        # Synthetic visualization placeholder
        canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 40
        cv2.putText(canvas, f"Camera {selected_camera} Live Gondola Stream", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        st.image(canvas, caption=f"Camera {selected_camera} Preview", use_container_width=True)

with d_col:
    st.subheader("📋 Priority Action Items for Staff")

    if not image_areas:
        st.info("No active area records found for this camera feed.")
    else:
        for area_id, data in image_areas.items():
            is_comp = data.get("planogram_compliant", True)
            stock_lvl = data.get("stock_level", "full").upper()
            stock_val = data.get("stock_value", 1.0)

            if not is_comp:
                st.error(f"🚨 **Area {area_id}** — **MISPLACED PRODUCT**\nRequires staff to remove wrong items.")
            elif stock_val == 0.0:
                st.warning(f"⚠️ **Area {area_id}** — **EMPTY SHELF (STOCK: 0%)**\nNeeds immediate replenishment.")
            elif stock_val == 0.5:
                st.info(f"🟡 **Area {area_id}** — **HALF DEPLETED (STOCK: 50%)**\nQueue for next restocking round.")
            else:
                st.success(f"✅ **Area {area_id}** — **COMPLIANT & FULL (STOCK: 100%)**")

st.divider()
st.subheader("📊 Detailed Prediction Records")
table_rows = []
for aid, val in image_areas.items():
    table_rows.append({
        "Area ID": aid,
        "Compliant": "✅ Yes" if val.get("planogram_compliant") else "❌ No",
        "Stock Level": val.get("stock_level", "").title(),
        "Stock Numeric Value": val.get("stock_value"),
    })

st.dataframe(table_rows, use_container_width=True)
