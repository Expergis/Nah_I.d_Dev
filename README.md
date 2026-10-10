# 🍌 Fresh Eyes on Fresh Produce — Shelf Monitoring Solution
> Autonomous shelf monitoring solution built for the **Sensei Challenge** at **JunctionX Lisbon 2026**.

## 📌 Problem & Overview
Fruits and vegetables in autonomous stores are sold loose, piled, unstandardized, and change appearance over time.
This system leverages ceiling camera imagery and store planograms (`planogram.json`) to autonomously:
1. **Detect & verify products** against the expected planogram (`planogram_compliant`).
2. **Estimate stock levels** (`empty`, `half`, `full` / `0.0`, `0.5`, `1.0`).
3. **Score with conservative loss-minimization**: Specifically optimized for the challenge's asymmetric penalty matrix (overestimating stock carries a severe −3 point penalty).

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
make setup
# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure API Keys
Edit `.env` and configure:
```env
GEMINI_API_KEY=your_key_here
VLM_PROVIDER=gemini
```

### 3. Run Pipeline
```bash
python3 scripts/run_pipeline.py --data-dir data/validation --output output/predictions.json
```

### 4. Evaluate Score
```bash
python3 scripts/validate.py --predictions output/predictions.json --annotations data/annotations.json
```

### 5. Launch Dashboard (UI/UX)
```bash
streamlit run app.py
```

---

## 🏗️ Architecture & Modules
- `src/planogram.py`: Parses quadrilateral planogram coordinate areas per camera. Handles edge boundary overflows.
- `src/image_processor.py`: Perspective transformation engine with padding support for negative camera coordinates.
- `src/vision_engine.py`: Multi-provider VLM pipeline with response caching, retry backoffs, and conservative uncertainty calibration.
- `src/prompt_templates.py`: Domain-specific prompts with Portuguese grocery taxonomy and scoring guardrails.
- `src/scoring.py`: Exact reproduction of the challenge's official points matrix.
- `app.py`: Real-time and pre-cached Streamlit dashboard designed for store operators.
