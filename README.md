# ML Prediction of Semiconductor Band Gaps from Composition

A machine-learning pipeline that predicts semiconductor band gaps from chemical
composition. The descriptor-driven approach is adapted from the elastic-constants
framework of Linton & Aidhy, *APL Mach. Learn.* **1**, 016109 (2023): instead of
predicting elastic constants of metal alloys from bond/element descriptors, we
predict an electronic property (band gap) of semiconductors from composition-based
elemental descriptors.

## Results so far (proof-of-concept)

On a curated set of 74 real semiconductors (experimental band gaps, 0.15–6.2 eV)
using composition descriptors only, 5-fold cross-validated performance:

| Model              | CV R² | CV RMSE (eV) |
|--------------------|-------|--------------|
| Linear Regression  | 0.49  | 0.96         |
| Random Forest      | 0.51  | 0.94         |
| Gradient Boosting  | 0.39  | 1.05         |

Real signal at small scale. The next step is scaling up with the full Materials
Project dataset (code included). See `REPORT.md` for the full writeup.

## Layout

```
semiconductor_ml/
├── src/
│   ├── fetch_data.py             # pull semiconductors from Materials Project
│   ├── featurize.py              # formulas -> descriptors (Magpie or fallback)
│   ├── train_models.py           # train/compare LR, RF, GBR on featurized data
│   ├── run_literature_demo.py    # real demo on the 74-compound literature set
│   ├── run_demo.py               # offline synthetic check (verifies the code)
│   └── models.py                 # shared model definitions + evaluation
├── data/literature_semiconductors.csv   # curated real band gaps
├── figures/    results/    requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
```

**1. Reproduce the proof-of-concept (no API key, runs immediately):**
```bash
python src/run_literature_demo.py
```
Writes `results/literature_metrics.json` and `figures/lit_*.png`.

**2. Scale up with your own Materials Project data:**
Get a free key at <https://next-gen.materialsproject.org/api>, then:
```bash
export MP_API_KEY="your_key_here"
python src/fetch_data.py --max-ehull 0.05
python src/featurize.py --in data/semiconductors.csv --target band_gap
python src/train_models.py --in data/features.csv --target band_gap
```

**3. Pure code sanity check (synthetic data, not results):**
```bash
python src/run_demo.py
```

## Method

- **Data:** experimental literature gaps (demo) / Materials Project PBE gaps (full).
- **Descriptors:** composition-weighted statistics of elemental properties.
- **Models:** Linear Regression, Random Forest, Gradient Boosting (the reference
  paper's family), scored by R² and RMSE with cross-validation and parity plots.

## Honesty note

Every number here comes from running the pipeline on real data; the synthetic
`run_demo.py` outputs are labeled as such and are not reported as findings.
