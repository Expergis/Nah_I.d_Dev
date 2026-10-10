# SENSEI PITCH — 3 min + 5 min QA

> **JunctionX Lisbon 2026 · Sensei Challenge**
> Team: [names] · Repo: github.com/Expergis/Nah_I.d_Dev

---

## TIMING BUDGET (3:00 total)

| Segment | Time | What |
|---|---|---|
| Problem + Insight | 0:00–0:40 | Why produce monitoring is hard, what we noticed |
| Demo | 0:40–2:00 | Live or pre-cached: image → crops → predictions → action list |
| How it works + Cost | 2:00–2:50 | Architecture, EV rule, **~0.1 cent per image** |
| Close | 2:50–3:00 | One-line value prop |

---

## SCRIPT

### [0:00–0:40] Problem + Insight

**"Fresh produce is the hardest part of a grocery store to monitor.**

Bananas, oranges, watermelons — they're loose, irregular, they occlude
each other. Today, a human walks the store every few hours to spot empty
shelves and misplaced products. At Sensei's scale, that doesn't scale.

**But here's the insight:** the store already has ceiling cameras. And
Sensei already has the planogram — exactly where each product *should* be.
We don't need AI to rediscover the shelf. We need AI to inspect each known
area and tell us: is it stocked, is it the right product, and what should
staff do next."

### [0:40–2:00] DEMO (80 seconds)

> **Use the Streamlit app or the pre-cached `demo/demo_predictions.json`.**
> If live inference is ready, run it. If not, use the cached demo.

**Walk through:**

1. Show the raw ceiling-camera image (camera 6215, bananas + oranges)
2. Overlay the 10 planogram polygons — "these are the known areas"
3. Click an area → show the perspective-corrected crop
4. Reveal the prediction:
   - *"Madeira Bananas — **Full** — Compliant ✅"*
   - *"Cat I Bananas — **Empty** — Restock needed ⚠️"*
   - *"Algarve Oranges — **Full** — Compliant ✅"*
5. Show the **prioritized action list**:
   - 🔴 Restock: 2 areas empty
   - 🟡 Check: 1 area mismatch
   - 🟢 OK: 7 areas fine
6. *"All of this from a single image, zero manual annotation."*

### [2:00–2:50] How it works + Cost

**"Here's how it works under the hood:**

**One.** We perspective-correct each planogram quadrilateral into a
rectified crop. This handles the ceiling-camera angle and gives the model
a clean, shelf-aligned view.

**Two.** A local vision-language model, Qwen3-VL-8B on a single 8GB
GPU, looks at each crop and says which of this camera's products it
sees and how much of the shelf is covered. We deliberately *don't* tell
it which product is expected there. When we did, it reported the
expected product on bare shelves. We compare with the planogram
afterwards, at the level Sensei says matters: a banana is a banana.

**Three.** We use a **risk-sensitive decision rule** calibrated to Sensei's
exact scoring matrix. Overestimating stock costs more than
underestimating, so our EV rule automatically biases conservative where
it matters — without a blanket 'always say empty' hack.

**And the cost: about a tenth of a cent per image.** Three choices keep
it low. A cheap colour-and-texture check skips the model entirely on bare
shelves, about a third of all areas. We use a non-thinking model with
one call per area; together those made us almost four times faster.
And it all runs on one 300-euro consumer GPU: 29 seconds and 2
watt-hours per image. That's 0.03 cents of electricity, and under a
tenth of a cent with the hardware included, because one box can serve
nine stores. That's on par with the cheapest cloud API, with no per-call
fees, and the images never leave the store."

> *Numbers: `scripts/estimate_cost.py`. After the full run, replace
> 29 s with the real average (total seconds / 181 images).*

### [2:50–3:00] Close

**"We turned fixed ceiling cameras into a continuously updated
inventory assistant for about a tenth of a cent per image. No manual
annotation. No cloud dependency.
Just geometry, a local model, and decisions calibrated to what actually
matters on the shelf."**

---

## QA PREP (5 minutes expected questions)

### "How accurate is it?"
> Honestly, we don't know our score yet. Only the hidden validation set
> has labels. What we did: checked the predictions by eye against the
> images on several cameras (bananas and oranges, peppers and aubergines,
> potatoes and onions). Every bare shelf, misplaced product and full
> basket we checked came out right, except for a few near-identical
> varieties. And the decision rule is built on the exact scoring: when
> the model is unsure, it picks the answer that loses the fewest points.
> [If we have a validation score by pitch day: "We scored X out of 578."]

### "What about occlusion — hands, shoppers?"
> The model reports whether a hand, person or cart hides a large part of
> the area. When it does, we soften its stock estimate toward "unsure",
> and the scoring rule then leans to the cheaper mistake (calling it
> lower). Areas that stick out of the camera frame (negative coordinates,
> like the top shelf of camera 6383) are filled with grey so the model
> never mistakes off-image pixels for an empty shelf.

### "What does it cost per image, exactly?"
> Measured: 29 s per image on one Radeon RX 9060 XT (8 GB, 150 W board
> power), Qwen3-VL-8B through Ollama. The whole PC draws about 250 W at the
> wall, so about 2 Wh per image, or €0.0003 at Portuguese business electricity
> rates (~€0.15/kWh). Hardware: a ~€900 box written off over 3 years.
> A store with 14 cameras taking one image every ~65 min makes ~310
> images a day. One box can do ~3,000, so it serves ~9 stores. All-in:
> **~€0.0006 per image (0.06 cents)**, or ~€0.003 (0.3 cents) if a store
> gets its own box. That's about €70 a year per store. Every assumption is a
> flag in `scripts/estimate_cost.py`.

### "How did you make it efficient?"
> 1. **Don't call the model when you don't need to.** A colour + texture
>    check (dark, unsaturated, smooth = bare shelf) skips ~30% of areas in
>    under a millisecond. It is deliberately conservative: it never fired
>    on full baskets of aubergines, avocados or cucumbers.
> 2. **No thinking tokens, one sample.** Switching to the non-thinking
>    instruct model and skipping the bare shelves made us ~3.8× faster
>    (102 s → 29 s per image). We also dropped 3-vote sampling to 1 call
>    per area.
> 3. **Small model, consumer GPU.** 8B parameters, quantized, fits in
>    8 GB VRAM. No datacenter GPU needed.
> 4. **Replayable decisions.** Raw model answers are cached, so we retune
>    thresholds and rules without paying for inference again.

### "Why not just use a cloud API?"
> It's not about price. For the same calls, Gemini 2.5 Flash-Lite would be
> ~€0.0007 per image, about the same as us. Gemini 2.5 Flash would be ~€0.0025,
> plus thinking tokens. We chose local because the images show shoppers and
> never leave the store, there are no per-call fees or rate limits, and it
> works offline. The API is still a one-line config switch
> (`SENSEI_BACKEND=gemini`) if Sensei prefers it.

### "How would you cut cost further?"
> Skip areas that haven't changed since the last frame (cameras are
> fixed, so a near-identical crop can reuse its answer). Try the 4B model
> where accuracy holds up. Drop the context image to roughly halve the
> image tokens per call.

### "What happens when the model is wrong?"
> The scoring is asymmetric: overestimating costs −3, underestimating
> −2. Our EV rule accounts for this. Under maximum uncertainty, it
> defaults to "empty" — the safest call. And we never skip an area:
> a missing prediction costs −5, so our fallback always emits something.

### "Can this handle other product types?"
> Yes, nothing is trained per product. The model picks from the product
> names listed for that camera in the planogram, so a new store or a new
> SKU only needs the planogram. Compliance is checked at product-family
> level (banana vs orange), with visually distinct varieties kept
> separate (green vs red pepper, red onion vs white).

### "What's your biggest limitation?"
> Two things. First, the half/full boundary is subjective: a dense pile of
> small tomatoes looks fuller than a few big watermelons, and we have no
> labels to calibrate it. Second, near-identical varieties (two banana
> grades, bagged potatoes vs bagged onions, shallots vs red onions): the
> model can't reliably tell them apart from a ceiling camera, so we only
> flag a mismatch across clearly different products.

### "Why Qwen3-VL specifically?"
> It's one of the strongest open-weights vision models that fits in 8GB
> of GPU memory: 8B parameters, 4-bit, structured JSON output. We use
> the non-thinking "instruct" version: the thinking version spent ~120
> extra tokens per answer for no gain on this task.

---

## SLIDES / VISUALS (if using a screen)

1. **Title slide**: Team name + "Fresh Eyes on Fresh Produce"
2. **Problem**: Photo of a produce aisle + "Who checks these shelves?"
3. **Demo**: Live Streamlit or pre-cached walkthrough
4. **Architecture**: Simple 3-box diagram (Image → Geometry + VLM → Decision)
5. **Cost**: "~0.1 cent per image" + 3 levers (skip bare shelves, no thinking, consumer GPU) + small table: local all-in vs Gemini Flash-Lite vs Flash
6. **Close**: One-liner + repo link

---

## CHECKLIST (before pitch)

- [ ] Demo runs without internet (local models + cached predictions)
- [ ] `demo/demo_predictions.json` loaded as fallback
- [ ] Streamlit app launches in < 10 sec
- [ ] Cost numbers re-run with the real average: `python scripts/estimate_cost.py --sec-per-image <avg> --debug <predictions.debug.json>`
- [ ] Pitch rehearsed once, timed with a phone
- [ ] Backup: 30-sec video recording of the demo on a laptop
- [ ] Repo link tested: github.com/Expergis/Nah_I.d_Dev (branch: dev)
