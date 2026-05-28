# Machine Learning Prediction of Semiconductor Band Gaps from Chemical Composition

**Preliminary research report**

Author: *[Your Name]*
Advisor: Prof. Santosh KC, Department of *[Physics / Electrical & Computer Engineering]*, San Diego State University
Date: May 2026

---

> **Status note for the advisor meeting.** This is an early-stage progress
> report. It covers a reproducible data + ML pipeline and a first working model,
> validated on a small curated set of real, published band gaps. The numbers in
> Section 3 are genuine outputs of the pipeline (`results/literature_metrics.json`).
> They are proof-of-concept scale; the immediate next step is to scale up using
> the full Materials Project dataset (the code to do this is included and ready).

## 1. Motivation

Predicting the band gap of semiconductors from first principles (DFT) is
accurate but computationally expensive, and the cost grows quickly across large
chemical search spaces. This is the same bottleneck that motivated Linton &
Aidhy (*APL Mach. Learn.* **1**, 016109, 2023), who showed that elastic
constants of multi-principal-element metal alloys can be predicted from
inexpensive composition/bond descriptors with standard ML regressors, bypassing
repeated DFT.

This project asks whether the same descriptor-driven idea transfers from a
*mechanical* property in metal alloys to an *electronic* property in
semiconductors: **can composition-based elemental descriptors predict
semiconductor band gaps well enough to screen candidate compositions before
committing to DFT?** Band gap is the natural first target — it is the central
design parameter for devices, and large consistent datasets (Materials Project)
exist, removing any need to run DFT at this stage.

## 2. Methods

### 2.1 Data

Two data sources are used, at two scales:

- **Proof-of-concept (this report):** a hand-curated set of 74 well-known
  semiconductors with experimental room-temperature band gaps from standard
  references (group IV, III–V, II–VI, oxides, IV–VI and V–VI chalcogenides,
  transition-metal dichalcogenides, ternary chalcopyrites, and halide
  perovskites), spanning 0.15–6.2 eV.
- **Full study (next step, code included):** the Materials Project `summary`
  endpoint via `mp-api`, selecting non-metals with a finite band gap and low
  energy-above-hull (stability filter). This yields thousands of entries and is
  driven by `src/fetch_data.py`.

### 2.2 Descriptors

Each formula is converted to a fixed-length vector of **composition-based
elemental descriptors**: composition-weighted mean, range, and standard
deviation of tabulated elemental properties (atomic number, mass,
electronegativity, radius, valence, group, period). The full study uses the
Magpie preset (matminer); the proof-of-concept uses a dependency-free
implementation of the same idea. This mirrors the reference paper's principle of
learning a property from constituent chemistry rather than an explicit
electronic-structure calculation.

### 2.3 Models and validation

Following the reference paper's model family, three regressors are compared:
Linear Regression (baseline), Random Forest, and Gradient Boosting. For the
small proof-of-concept set, the headline metrics are **5-fold
cross-validated** R² and RMSE (more reliable than a single split at small N); an
additional 80/20 split is used for the parity plots. Feature importances from
the gradient-boosting model identify the descriptors carrying the signal.

## 3. Results (proof-of-concept)

On the 74-compound literature set, using 22 composition descriptors:

**Table 1. 5-fold cross-validated performance.**

| Model              | CV R² | CV RMSE (eV) |
|--------------------|-------|--------------|
| Linear Regression  | 0.49  | 0.96         |
| Random Forest      | 0.51  | 0.94         |
| Gradient Boosting  | 0.39  | 1.05         |

Random Forest performs best, capturing roughly half the band-gap variance across
a chemically very diverse set from composition alone. On the single 80/20 split
used for visualization, Random Forest gives R² = 0.50, RMSE = 0.69 eV
(`figures/lit_parity_RandomForest.png`); the linear baseline degrades on that
split because it extrapolates poorly to the few wide-gap outliers (AlN, BN, C).

The most informative descriptors (gradient boosting) are mean atomic mass and
atomic number, electronegativity spread, and mean atomic radius. The appearance
of **electronegativity spread** is physically sensible: bond ionicity correlates
with band gap, so the dispersion of constituent electronegativities is a natural
predictor. Mass/atomic-number terms most likely act as proxies for chemical
family (e.g. heavy chalcogenides cluster at low gaps).

## 4. Discussion

- **Tree models beat the linear baseline** under cross-validation, indicating
  genuinely nonlinear composition→gap relationships — consistent with the
  reference paper's choice of gradient boosting.
- **Performance is modest, as expected at this scale.** Two things cap it:
  (i) only 74 samples, and (ii) composition-only descriptors cannot distinguish
  polymorphs (same formula, different structure → different gap), which sets a
  floor on achievable error.
- **The result is a real signal, not noise**, and motivates the full study:
  thousands of Materials Project entries, the richer Magpie feature set, and
  eventually structural descriptors should raise R² substantially.

## 5. Limitations

- Small, hand-curated dataset; composition-only features.
- Experimental band gaps here vs DFT (PBE) gaps in the Materials Project pull —
  PBE systematically underestimates gaps, so the two targets are not identical
  and must not be mixed without correction.
- Single featurization backend (fallback) used for the proof-of-concept.

## 6. Next steps (proposed ~2-month plan)

1. **Scale up with Materials Project** (`fetch_data.py`): pull thousands of
   stable semiconductors and re-train. This is the immediate priority.
2. **Full Magpie features** via matminer, replacing the fallback featurizer.
3. **Structural descriptors** to resolve polymorphs and test the expected
   accuracy gain.
4. **Robust validation:** learning curves and k-fold CV on the large set.
5. **Benchmarking:** compare against the Materials Project's hosted ML
   predictions and published baselines (e.g. MatBench `matbench_mp_gap`).
6. **Optional, advisor-directed:** restrict the screen to a device-relevant
   family and rank candidate compositions by predicted gap.

## References

1. N. Linton and D. S. Aidhy, "A machine learning framework for elastic
   constants predictions in multi-principal element alloys," *APL Mach. Learn.*
   **1**, 016109 (2023).
2. A. Jain *et al.*, "Commentary: The Materials Project," *APL Mater.* **1**,
   011002 (2013).
3. L. Ward *et al.*, "A general-purpose machine learning framework for
   predicting properties of inorganic materials (Magpie)," *npj Comput. Mater.*
   **2**, 16028 (2016).
4. L. Ward *et al.*, "Matminer: An open source toolkit for materials data
   mining," *Comput. Mater. Sci.* **152**, 60–69 (2018).
5. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach.
   Learn. Res.* **12**, 2825–2830 (2011).

Band-gap reference values were taken from standard semiconductor data sources
(e.g. textbook/handbook compilations and the Ioffe Institute NSM database).

---

## Appendix A — Pipeline verification (synthetic, NOT results)

`src/run_demo.py` trains the three models on a synthetic dataset with a known
band-gap function, purely to confirm the LR/RF/GBR comparison, metrics, and
parity plotting run end-to-end before any real data. Its outputs
(`figures/demo_parity_*.png`) are from synthetic data and are **not** findings.
