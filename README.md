# Sensei — Fruits & Vegetables Shelf Monitoring

JunctionX Lisbon 2026 · Sensei challenge: *Fresh eyes on fresh produce.*

For every planogram area in every ceiling-camera image, we predict whether
the product shown is the one the planogram expects, and whether the area
is empty, half or full. There is no manual annotation and no training: a
local vision-language model inspects each area, and a decision rule tuned
to the challenge's scoring turns its answers into predictions.

## The pipeline that produces our submission: `run_local.py`

One self-contained script (Python + OpenCV + a local Ollama server).
Per area:

1. **Rectify.** Warp the planogram quadrilateral into a 512×384 crop.
   Parts outside the camera image become grey.
2. **Bare-shelf shortcut.** If the crop is dark (`dark > 0.60`), has almost
   no saturated colour (`fruit < 0.06`) and is smooth (`texture < 0.04`,
   Canny edge density), call it *empty + compliant* without asking the
   model. Deliberately conservative: it never fired on full baskets of
   aubergine, avocado or cucumber in our checks.
3. **Ask the model, un-primed.** Qwen3-VL-8B-Instruct (4-bit, via Ollama)
   gets the crop plus a small full-camera view with the area outlined. It
   is told only which products exist on *this camera*, not which one is
   expected here (telling it made it "see" the expected product on bare
   shelves). It returns JSON: `product` (from the list, `empty` or
   `other`), `coverage_pct` (0–100) and `hand_occluding`.
4. **Decide.**
   - *Stock*: coverage → probabilities over empty/half/full → pick the
     level with the best expected score under the official points
     (correct +1, too low −2, too high −3). Uncertainty leans low.
   - *Compliance*: the product the model saw is compared with the
     expected one at **family** level (`BANANA CAT I` = `BANANA DA
     MADEIRA`), because the challenge says products are recognised
     generically. Visually distinct varieties are kept apart (green vs red
     peppers, red vs white grapes, red onion, broccoli vs cabbage…).
     `empty` and `other` never count as a mismatch.
5. **Never leave a hole.** A failed call falls back to *empty + compliant*
   (a missing area costs −5). The output is validated so every planogram
   area of every image is present.

Every raw model answer is cached (`.cache_local/`), so re-running with
different decision settings recomputes predictions in seconds without new
model calls. `predictions.debug.json` keeps each area's features and raw
answers.

### Exact settings used for the submission

```powershell
# Ollama on an AMD Radeon GPU (Windows). Two requests in parallel keep the GPU busy.
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen_amd.ps1 -Model qwen3-vl:4b-instruct
$env:OLLAMA_NUM_PARALLEL = "2"; ollama serve          # in its own window

python run_local.py --model qwen3-vl:4b-instruct --num-ctx 3072 --workers 2 `
    --images-dir <folder with one sub-folder per camera id> --out predictions.json
```

Model choice: Qwen3-VL **4B** instruct. On 152 hand-labelled areas, scored with the
official points (/304): 2B −92, **4B 181**, 8B 177, 30B-A3B 160 (`scripts/compare_models.py`).

Defaults: 1 call per area at temperature 0, prompt version `v2`,
`--dark-thr 0.60 --fruit-max 0.06 --texture-max 0.04`, `--cov-offset 0`,
context view on. `--num-ctx` must be at least 3072 (a request is about
2,500 tokens). Images are found as `<images-dir>/<camera_id>/<timestamp>.jpg`;
a `<camera_id>` folder at any depth below `<images-dir>` also works.
Measured speed: 38 s per image on an RX 9060 XT 8 GB (2 workers).

Useful flags: `--camera 6215 --image 1751001268` (smoke test),
`--verbose` (Ollama timing per call), `--save-crops crops` (dump crops to
eyeball), `--verify-mismatch` (second call to confirm non-compliance).

### Cost per image

`python scripts/estimate_cost.py --sec-per-image <measured> --debug predictions.debug.json`
turns a measured run time into € per image (energy + hardware) and
compares it with cloud APIs for the same calls.

## Repository layout

```
run_local.py            # THE pipeline (identical to the copy that ran the submission)
planogram.json          # Sensei planogram (14 cameras, 152 areas)
scripts/
  setup_qwen_amd.ps1    # Ollama + model setup on the AMD/Windows box
  estimate_cost.py      # cost per analysed image (pitch numbers)
  contact_sheet.py      # labelled crop grids for checking by eye
  explore_data.py       # dataset summary
sensei/                 # EARLIER design, not used for the submission (see below)
tests/                  # unit tests for sensei/ (geometry, decision, output)

```

### About `sensei/`

`sensei/` is our first, more modular design: compliance asked as a
yes/no question naming the expected product, 3 sampled votes per area,
first-token probabilities, an optional Gemini backend and a planned DINOv2
stock signal. We replaced it with `run_local.py` after finding that naming
the expected product made the model report it on bare shelves. Its
colour/texture shortcut and family matching were kept in sync with
`run_local.py` and are covered by the tests (`pytest tests`), but the
submission does not run this package.
