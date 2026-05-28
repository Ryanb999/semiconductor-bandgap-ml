# Talking points — meeting with Prof. Santosh KC

A one-page guide for walking through the project. Keep it honest and short;
let the repo and the parity plot do the work.

## The one-sentence pitch
"I built a reproducible pipeline that predicts semiconductor band gaps from
composition alone — extending the descriptor-based ML idea from the Linton–Aidhy
alloy elastic-constants paper to an electronic property — and validated it on a
set of real semiconductors as a proof of concept."

## What to show, in order
1. **The repo** — README first (explains the approach and shows the results table).
2. **The parity plot** — `figures/lit_parity_RandomForest.png`: predicted vs
   experimental band gap, points scattered around the diagonal.
3. **REPORT.md** — the short writeup: motivation, methods, results, next steps.
4. **The code** — `fetch_data.py` (Materials Project query) and `train_models.py`,
   to show it's a real, runnable pipeline, not a one-off notebook.

## What I did (be specific)
- Curated 74 real semiconductors with literature band gaps (0.15–6.2 eV).
- Built composition descriptors (weighted mean/range/std of elemental properties).
- Compared Linear Regression, Random Forest, Gradient Boosting with 5-fold CV.
- Wrote the full Materials Project fetch pipeline so the study can scale to
  thousands of compounds.

## The honest headline result
- Random Forest: cross-validated R² ≈ 0.51, RMSE ≈ 0.94 eV from composition only.
- Real signal, modest accuracy — exactly what's expected at 74 samples with no
  structural information.

## Why it's modest (get ahead of the obvious question)
- Small dataset (74).
- Composition-only features can't separate polymorphs (same formula, different
  structure, different gap) — a hard floor on accuracy until structure is added.
- PBE band gaps (Materials Project) underestimate experiment; the two targets
  shouldn't be mixed without a correction.

## What I'd do next (the actual plan)
1. Scale up: pull thousands of stable semiconductors from Materials Project.
2. Switch to full Magpie features (matminer).
3. Add structural descriptors to resolve polymorphs.
4. Benchmark against MatBench `matbench_mp_gap` and MP's hosted ML.

## Questions to ask the advisor
- Is there a specific materials family the group cares about that I should focus
  the screen on?
- Band gap, or would a different target (formation energy, a mechanical property)
  be more useful to the group's work?
- Any preferred validation/benchmarking standards for publishing this kind of result?
