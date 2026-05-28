"""
train_models.py
---------------
Train and compare LR / RandomForest / GradientBoosting on the featurized
semiconductor dataset, then write:
  - results/metrics.json         (R^2 and RMSE per model)
  - figures/parity_<model>.png   (DFT/MP value vs ML prediction)
  - figures/feature_importance.png (top descriptors, tree models)

Usage
-----
    python src/train_models.py --in data/features.csv --target band_gap
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import train_and_evaluate, RANDOM_STATE


def parity_plot(y_true, y_pred, name, r2, rmse, target, outpath):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(y_true, y_pred, s=22, facecolors="none", edgecolors="tab:green", linewidths=1.1)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Reference {target} (eV)")
    ax.set_ylabel(f"ML predicted {target} (eV)")
    ax.set_title(name)
    ax.text(0.05, 0.92, f"$R^2$={r2:.2f}\nRMSE={rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def importance_plot(model, feature_names, outpath, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(range(len(order)), imp[order][::-1], color="tab:blue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("Gradient Boosting feature importance")
    ax.set_title("Top descriptors")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Train and compare ML models.")
    p.add_argument("--in", dest="infile", default="data/features.csv")
    p.add_argument("--target", default="band_gap")
    p.add_argument("--figdir", default="figures")
    p.add_argument("--resultsdir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)

    df = pd.read_csv(args.infile)
    y = df[args.target].values
    X_df = df.drop(columns=[args.target])
    feature_names = list(X_df.columns)
    X = X_df.values

    out = train_and_evaluate(X, y)
    results = out["results"]

    print(f"Dataset: {len(y)} samples, {X.shape[1]} descriptors "
          f"(train={out['n_train']}, test={out['n_test']})")
    print(f"{'Model':<20}{'R^2':>8}{'RMSE (eV)':>12}")
    for name, m in results.items():
        print(f"{name:<20}{m['R2']:>8.3f}{m['RMSE']:>12.3f}")

    # Parity plots
    for name in results:
        parity_plot(out["y_test"], out["predictions"][name], name,
                    results[name]["R2"], results[name]["RMSE"], args.target,
                    os.path.join(args.figdir, f"parity_{name}.png"))

    # Feature importance from the gradient boosting model
    importance_plot(out["fitted"]["GradientBoosting"], feature_names,
                    os.path.join(args.figdir, "feature_importance.png"))

    payload = {
        "target": args.target,
        "n_samples": int(len(y)),
        "n_descriptors": int(X.shape[1]),
        "n_train": out["n_train"],
        "n_test": out["n_test"],
        "random_state": RANDOM_STATE,
        "models": results,
        "data_source": "REAL (Materials Project)" if "features.csv" in args.infile
                       else "see input file",
    }
    with open(os.path.join(args.resultsdir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {os.path.join(args.resultsdir, 'metrics.json')} and figures to {args.figdir}/")


if __name__ == "__main__":
    main()
