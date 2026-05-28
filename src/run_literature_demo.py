"""
run_literature_demo.py
----------------------
Runs the REAL pipeline (fallback featurizer + LR/RF/GBR) on a small CURATED
dataset of published, experimental semiconductor band gaps
(data/literature_semiconductors.csv).

This is a genuine result on real data -- but the dataset is small (~74
compounds) and hand-curated, so treat it as a proof-of-concept. The full study
uses thousands of entries pulled from the Materials Project via fetch_data.py.

Reports BOTH:
  - 5-fold cross-validated R^2 and RMSE (robust estimate for small N)
  - a single 80/20 split (for parity plots, as in the reference paper)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold

from models import get_models, train_and_evaluate, evaluate, RANDOM_STATE
from featurize import fallback_features

TARGET = "band_gap"
N_SPLITS = 5


def main():
    df = pd.read_csv("data/literature_semiconductors.csv")
    feats = fallback_features(df["formula_pretty"])
    y = df.loc[feats.index, TARGET].values
    X = feats.values
    feature_names = list(feats.columns)

    print("=" * 70)
    print(f" LITERATURE DEMO  |  real data, small curated set (n={len(y)})")
    print("=" * 70)
    print(f"{X.shape[1]} composition descriptors | gap range "
          f"{y.min():.2f}-{y.max():.2f} eV\n")

    # ---- 5-fold cross-validation (headline numbers) ----
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print(f"{'Model':<20}{'CV R^2':>10}{'CV RMSE (eV)':>15}")
    for name, model in get_models().items():
        y_cv = cross_val_predict(model, X, y, cv=kf)
        r2, rmse = evaluate(y, y_cv)
        cv_results[name] = {"R2": round(r2, 3), "RMSE_eV": round(rmse, 3)}
        print(f"{name:<20}{r2:>10.3f}{rmse:>15.3f}")

    # ---- single 80/20 split (for parity plots) ----
    out = train_and_evaluate(X, y)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name in out["results"]:
        yp, yt = out["predictions"][name], out["y_test"]
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(yt, yp, s=30, facecolors="none", edgecolors="tab:green", linewidths=1.2)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Experimental band gap (eV)")
        ax.set_ylabel("ML predicted gap (eV)")
        ax.set_title(f"Literature demo -- {name}")
        ax.text(0.05, 0.92,
                f"$R^2$={out['results'][name]['R2']:.2f}\nRMSE={out['results'][name]['RMSE']:.2f} eV",
                transform=ax.transAxes, va="top", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"figures/lit_parity_{name}.png", dpi=160)
        plt.close(fig)

    # ---- feature importance (GBR) ----
    gbr = out["fitted"]["GradientBoosting"]
    imp = gbr.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(order)), imp[order][::-1], color="tab:blue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Gradient Boosting feature importance")
    ax.set_title("Top descriptors (literature demo)")
    fig.tight_layout()
    fig.savefig("figures/lit_feature_importance.png", dpi=160)
    plt.close(fig)

    with open("results/literature_metrics.json", "w") as fh:
        json.dump({
            "note": "Real data, small curated literature dataset. Proof-of-concept scale.",
            "n_samples": int(len(y)),
            "n_descriptors": int(X.shape[1]),
            "gap_range_eV": [float(y.min()), float(y.max())],
            "cross_validation_5fold": cv_results,
            "single_split_80_20": out["results"],
            "top_descriptors": [feature_names[i] for i in order],
        }, fh, indent=2)

    print("\nSingle 80/20 split (parity plots):")
    for name, m in out["results"].items():
        print(f"  {name:<20} R^2={m['R2']:.3f}  RMSE={m['RMSE']:.3f} eV")
    print("\nFigures -> figures/lit_*.png   Metrics -> results/literature_metrics.json")
    print(f"Top descriptors: {', '.join(feature_names[i] for i in order[:5])}")


if __name__ == "__main__":
    main()
