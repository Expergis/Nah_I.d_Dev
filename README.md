# Sensei — Fruits & Vegetables Shelf Monitoring

JunctionX Lisbon 2026 · Sensei challenge.

Detect, count and check loose produce on each gondola area against the
planogram, from ceiling-camera images — with **zero manual annotation**.

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
