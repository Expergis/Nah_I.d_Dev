# Sensei — Fruits & Vegetables Shelf Monitoring

JunctionX Lisbon 2026 · Sensei challenge.

Detect, count and check loose produce on each gondola area against the
planogram, from ceiling-camera images — with **zero manual annotation**.

## Approach

The planogram gives us the hard part: exact per-area coordinates. So the
pipeline is:

1. **Geometry** — perspective-warp each planogram quadrilateral into a
   rectified crop, plus a visibility mask (off-image areas are flagged,
   never confused with empty shelves) and a wider context view.
2. **VLM inspection** — Gemini (structured JSON output) answers two
   questions per area: *compliance* (closed-set vs expected product) and
   *stock* (coverage % + empty/half/full). N samples per area = votes.
3. **Risk-sensitive decision** — the official scoring is asymmetric
   (over-calling stock costs −3, under-calling −2), so we pick the level
   with the highest **expected score** from the vote probabilities — not a
   blanket conservative bias.
4. **Guaranteed completeness** — the output skeleton is pre-filled for
   every planogram area; an API failure yields a documented fallback,
   never a missing area (missing = −5).

## Layout

```
sensei/           # pipeline package
  config.py       # env-driven config (SENSEI_*, GEMINI_API_KEY)
  planogram.py    # planogram.json loader
  geometry.py     # quad warp, visibility mask, context crop
  vlm.py          # async Gemini REST client (cache + retries)
  prompting.py    # per-area prompt builder
  decision.py     # expected-value decision rule
  pipeline.py     # orchestration
  output.py       # exact submission format + validator
scripts/
  run_pipeline.py # CLI entry point
  contact_sheet.py# labeled crop grids for manual audit
  audit_report.py # predictions vs manual audit labels
audit/            # team's manual labels (calibration set)
data/             # dataset (git-ignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
set GEMINI_API_KEY=...        # or put it in .env
```

## Usage

```bash
# sanity check without any API calls
python scripts/run_pipeline.py --dry-run

# one camera / one image (cheap iteration)
python scripts/run_pipeline.py --camera 6215 --image 1751052241.jpg

# full run -> outputs/predictions.json (exact submission format)
python scripts/run_pipeline.py
```

## Tests

```bash
python -m pytest tests/ -q
```

## Decision rule (official scoring)

| predict \ truth | empty | half | full |
|---|---|---|---|
| **empty** | +1 | −2 | −2 |
| **half** | −3 | +1 | −2 |
| **full** | −3 | −3 | +1 |

`EV(empty)=pE−2pH−2pF` · `EV(half)=−3pE+pH−2pF` · `EV(full)=−3pE−3pH+pF`

Compliance: predict compliant iff its vote share ≥ noncompliant's
(+1/−2 asymmetry → threshold 0.5; ties go to compliant, the base rate).

## Roadmap (hackathon hours)

- [x] skeleton, geometry, client, decision layer, validator
- [ ] first live end-to-end run on real data
- [ ] audit set labelling (~100 areas) + threshold calibration
- [ ] reference-image prompting (full/empty references per area)
- [ ] second-model routing on low-agreement areas
- [ ] temporal consistency checks across timestamps
- [ ] Streamlit staff view (colored quads + prioritized action list)
- [ ] 2-minute pitch
