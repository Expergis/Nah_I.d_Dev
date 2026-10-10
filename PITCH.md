# SENSEI PITCH — 3 min + 5 min QA

> **JunctionX Lisbon 2026 · Sensei Challenge**
> Team: [names] · Repo: github.com/Expergis/Nah_I.d_Dev

---

## TIMING BUDGET (3:00 total)

| Segment | Time | What |
|---|---|---|
| Problem + Insight | 0:00–0:40 | Why produce monitoring is hard, what we noticed |
| Demo | 0:40–2:00 | Live or pre-cached: image → crops → predictions → action list |
| How it works + Cost | 2:00–2:50 | Architecture, EV rule, **$0 API cost** |
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

**Two.** A local vision-language model — Qwen3-VL-8B, running on a single
8GB GPU — inspects each crop. It answers two questions: is the product
compliant with the planogram, and what's the stock level?

**Three.** We use a **risk-sensitive decision rule** calibrated to Sensei's
exact scoring matrix. Overestimating stock costs more than
underestimating, so our EV rule automatically biases conservative where
it matters — without a blanket 'always say empty' hack.

**And here's the part that matters for cost:**

**Our API cost per analyzed image is zero.** All inference runs locally on
one GPU. No tokens, no rate limits, no per-call pricing. At Sensei's
production scale — roughly 470,000 images per year across 14 cameras —
our marginal cost is electricity: about 70 euros a year. A comparable
cloud-API approach would cost between 1,200 and 6,000 dollars per year
in token fees alone."

### [2:50–3:00] Close

**"We turned fixed ceiling cameras into a continuously updated,
cost-free inventory assistant. No manual annotation. No API dependency.
Just geometry, a local model, and decisions calibrated to what actually
matters on the shelf."**

---

## QA PREP (5 minutes expected questions)

### "How accurate is it?"
> We're calibrating against a hand-labeled audit set of ~150 areas.
> The EV decision rule is mathematically optimal for the scoring matrix,
> so even with imperfect probability estimates, we're making the
> least-costly wrong call. [If we have a number by pitch day: "On our
> audit set, we're scoring X points out of 578 max."]

### "What about occlusion — hands, shoppers?"
> The prompt explicitly tells the model to ignore hands and estimate
> stock behind them. We also compute a visibility mask — if less than
> 90% of the area is in-frame, the model is told to say 'uncertain'
> rather than guess.

### "Why not just use a cloud API?"
> We can — it's a one-line config change (`SENSEI_BACKEND=gemini`).
> But for a hackathon judged on cost, local inference is the clear
> winner: $0 marginal cost, offline-capable, no rate limits. We kept
> the API path as a booster for ambiguous cases.

### "What happens when the model is wrong?"
> The scoring is asymmetric: overestimating costs −3, underestimating
> −2. Our EV rule accounts for this. Under maximum uncertainty, it
> defaults to "empty" — the safest call. And we never skip an area:
> a missing prediction costs −5, so our fallback always emits something.

### "Can this handle other product types?"
> Yes. The planogram defines the expected product per area. We ask the
> model "is this consistent with [product name]?" — not "what fruit is
> this?" That's a much easier question and generalizes to any SKU.

### "What's your biggest limitation?"
> The half/full boundary is subjective. A dense pile of small tomatoes
> looks fuller than a sparse pile of large watermelons. We're
> calibrating this against our audit labels and using DINOv2 embeddings
> as a second signal to sharpen that boundary.

### "Why Qwen3-VL specifically?"
> It's the best open-weights VLM that fits in 8GB VRAM. 8B parameters,
> Q4 quantized, handles image + text with structured JSON output.
> We benchmarked it against minicpm-v and gemma3 on our audit set —
> Qwen3-VL wins on both compliance and stock accuracy.

---

## SLIDES / VISUALS (if using a screen)

1. **Title slide**: Team name + "Fresh Eyes on Fresh Produce"
2. **Problem**: Photo of a produce aisle + "Who checks these shelves?"
3. **Demo**: Live Streamlit or pre-cached walkthrough
4. **Architecture**: Simple 3-box diagram (Image → Geometry + VLM → Decision)
5. **Cost**: Side-by-side table (Ours $0 vs API $1,200+/year)
6. **Close**: One-liner + repo link

---

## CHECKLIST (before pitch)

- [ ] Demo runs without internet (local models + cached predictions)
- [ ] `demo/demo_predictions.json` loaded as fallback
- [ ] Streamlit app launches in < 10 sec
- [ ] Cost numbers verified with `scripts/estimate_cost.py --api-compare`
- [ ] Pitch rehearsed once, timed with a phone
- [ ] Backup: 30-sec video recording of the demo on a laptop
- [ ] Repo link tested: github.com/Expergis/Nah_I.d_Dev (branch: dev)
