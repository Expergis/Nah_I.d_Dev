"""Streamlit Real-Time Shelf Monitor, Diagnostic Audit & Presentation Dashboard.

Run with:
    streamlit run scripts/dashboard.py
"""
import json
import math
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sensei Produce Intelligence",
    page_icon="🥑",
    layout="wide",
)

st.title("🛒 Sensei Shelf Intelligence — Continente Bom Dia (Leiria)")
st.markdown("Zero-shot autonomous produce monitoring, continuous-batching VLM diagnostics, and planogram auditing.")

PRED_PATH = Path("output/predictions.json")
CACHE_PATH = Path(".cache/vlm")

if not PRED_PATH.exists():
    st.warning("No predictions found in `output/predictions.json`. Run pipeline first!")
    st.stop()

with open(PRED_PATH) as f:
    predictions = json.load(f)

# Global metrics calculation
total_cameras = len(predictions)
total_areas = sum(len(areas) for cam in predictions.values() for areas in cam.values())
stock_counts = {"empty": 0, "half": 0, "full": 0}
compliant_count = 0

for cam in predictions.values():
    for img in cam.values():
        for area_id, data in img.items():
            lvl = data.get("stock_level", "unknown").lower()
            if lvl in stock_counts:
                stock_counts[lvl] += 1
            if data.get("planogram_compliant"):
                compliant_count += 1

# Calculate Entropy (prediction differentiation)
entropy = 0.0
for c in stock_counts.values():
    if c > 0 and total_areas > 0:
        p = c / total_areas
        entropy -= p * math.log2(p)

tabs = st.tabs(["📊 Executive Overview", "🔬 Deep Model Diagnostics & Optimizations", "📷 Camera Explorer & Alerts"])

# ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
with tabs[0]:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cameras Monitored", total_cameras)
    col2.metric("Shelf Slots Monitored", total_areas)
    col3.metric("Planogram Compliance", f"{(compliant_count/total_areas*100):.1f}%" if total_areas else "0%")
    col4.metric("Immediate Restocks", stock_counts["empty"])
    col5.metric("Model Shannon Entropy", f"{entropy:.2f} / 1.58", help="1.58 = perfectly balanced distribution across classes")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 Shelf Fill Distribution")
        df_stock = pd.DataFrame({
            "Stock Level": ["EMPTY (0-15%)", "HALF (15-70%)", "FULL (>70%)"],
            "Count": [stock_counts["empty"], stock_counts["half"], stock_counts["full"]]
        })
        st.bar_chart(df_stock.set_index("Stock Level"))

    with c2:
        st.subheader("🛡️ Asymmetric Risk Profile")
        st.markdown("""
        In the Sensei challenge, mistakes are asymmetric:
        * **Overcalling 'Full' when Half/Empty**: Costs **-3 points** (Severe Overstock Penalty).
        * **Undercalling 'Empty/Half' when Full**: Costs **-2 points** (Mild Penalty).
        * **Accurate Calling**: Gains **+1 point**.
        """)
        
        full_pct = (stock_counts["full"] / total_areas * 100) if total_areas else 0
        if full_pct > 50:
            st.warning(f"⚠️ {full_pct:.1f}% predicted as FULL. Consider increasing coverage threshold for 'full' to hedge against the -3 penalty.")
        else:
            st.success(f"✅ Conservative stock posture ({full_pct:.1f}% FULL). Maximizes Expected Value (EV).")

# ----------------- TAB 2: DIAGNOSTICS & OPTIMIZATION -----------------
with tabs[1]:
    st.subheader("🔬 Optimization Opportunities & Visual Reasoning Audit")
    
    # Audit Cache
    cached_reasons = []
    if CACHE_PATH.exists():
        for f in CACHE_PATH.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "reason" in data:
                    cached_reasons.append({
                        "Reasoning": data.get("reason"),
                        "Stock": data.get("stock_level"),
                        "Compliance": data.get("compliance"),
                        "Coverage %": data.get("coverage_percent"),
                    })
            except Exception:
                pass

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("### 💡 Recommended Tuning Actions")
        st.markdown("""
        1. **Crop Context Padding (`SENSEI_CONTEXT_PAD`)**:
           - Default is `0.35` (35% extra margin around shelf slot).
           - *Optimization*: If neighboring items bleed into crops, reduce to `0.20`.
        2. **Multi-Sample Bayesian Voting (`SENSEI_SAMPLES`)**:
           - Currently set to `1` for speed.
           - *Optimization*: Bump to `3` with temperature `0.6` to leverage `decision.py` Bayesian EV voting.
        3. **Batch Concurrency on RTX 4060**:
           - Set `SENSEI_CONCURRENCY=4` to match `llama-server --parallel 4`.
        """)

    with d2:
        st.markdown("### 🧠 Live Visual Rationale from Qwen2.5-VL")
        if cached_reasons:
            st.dataframe(pd.DataFrame(cached_reasons).head(10), use_container_width=True)
        else:
            st.info("No cached reasoning logs found yet. Run pipeline without `--no-cache` to view rationale logs.")

# ----------------- TAB 3: CAMERA EXPLORER -----------------
with tabs[2]:
    selected_cam = st.sidebar.selectbox("Select Camera", sorted(list(predictions.keys())))
    images_for_cam = predictions.get(selected_cam, {})
    selected_img = st.sidebar.selectbox("Select Timestamp", sorted(list(images_for_cam.keys())))

    st.subheader(f"Camera {selected_cam} — Image: `{selected_img}`")
    areas_pred = images_for_cam.get(selected_img, {})

    col_tbl, col_alert = st.columns([1, 1])

    with col_tbl:
        st.markdown("### 📋 Area Predictions")
        table_data = []
        for area_id, data in areas_pred.items():
            table_data.append({
                "Area ID": area_id,
                "Compliant": "✅" if data.get("planogram_compliant") else "❌",
                "Stock Level": data.get("stock_level", "").upper(),
                "Stock Value": f"{data.get('stock_value', 0.0) * 100:.0f}%",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    with col_alert:
        st.markdown("### 🚨 Store Associate Action Items")
        urgent = [a for a, d in areas_pred.items() if d.get("stock_level") == "empty" or not d.get("planogram_compliant")]
        if urgent:
            for u in urgent:
                d = areas_pred[u]
                reasons = []
                if d.get("stock_level") == "empty":
                    reasons.append("CRITICAL OUT-OF-STOCK")
                if not d.get("planogram_compliant"):
                    reasons.append("PRODUCT MISMATCH")
                st.error(f"**Slot {u}**: {' & '.join(reasons)}")
        else:
            st.success("All slots in this view are healthy and compliant!")
