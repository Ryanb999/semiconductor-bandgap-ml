"""
run_demo.py
-----------
SELF-CONTAINED DEMONSTRATION -- no network, no API key, no extra packages.

This generates a SYNTHETIC dataset (random compositions with a band gap defined
by a known nonlinear function of elemental descriptors plus noise), then runs
the exact same LR / RandomForest / GradientBoosting comparison the real
pipeline uses. Its only purpose is to prove the code works end-to-end and to
show you the output format.

  ##############################################################################
  #  THE NUMBERS AND PLOTS THIS PRODUCES ARE FROM SYNTHETIC DATA.              #
  #  THEY ARE NOT RESULTS. Do not put them in the paper as findings.          #
  #  Your real results come from running fetch_data.py -> featurize.py ->     #
  #  train_models.py on actual Materials Project data.                        #
  ##############################################################################

Usage
-----
    python src/run_demo.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import train_and_evaluate, RANDOM_STATE
from featurize import ELEM, PROP_NAMES

rng = np.random.default_rng(RANDOM_STATE)

# A small pool of elements that commonly appear in semiconductors.
POOL = ["Si", "Ge", "Ga", "As", "In", "Sb", "Zn", "S", "Se", "Te",
        "Cd", "Al", "P", "N", "O", "Sn", "C", "B"]


def random_compound():
    k = rng.integers(2, 4)  # 2 or 3 elements
    elems = list(rng.choice(POOL, size=k, replace=False))
    fracs = rng.dirichlet(np.ones(k))
    return elems, fracs


def descriptors(elems, fracs):
    mat = np.array([ELEM[e] for e in elems], dtype=float)
    w = np.asarray(fracs)
    feat = []
    for j in range(len(PROP_NAMES)):
        col = mat[:, j]
        mean = float(np.dot(w, col))
        feat.extend([mean, float(col.max() - col.min())])
    return np.array(feat)


def synthetic_gap(elems, fracs):
    """A deterministic but nonlinear 'true' band gap built from electronegativity
    spread and mean valence -- purely to give the models a learnable signal."""
    mat = np.array([ELEM[e] for e in elems], dtype=float)
    w = np.asarray(fracs)
    en = mat[:, PROP_NAMES.index("electroneg")]
    val = mat[:, PROP_NAMES.index("valence")]
    en_range = en.max() - en.min()
    val_mean = float(np.dot(w, val))
    gap = 0.9 * en_range + 0.15 * (val_mean - 4) ** 2 + 0.3
    return max(gap, 0.05)


def main():
    print("=" * 70)
    print(" DEMONSTRATION RUN ON SYNTHETIC DATA -- NOT REAL RESULTS")
    print("=" * 70)

    N = 600
    X, y = [], []
    for _ in range(N):
        elems, fracs = random_compound()
        X.append(descriptors(elems, fracs))
        y.append(synthetic_gap(elems, fracs) + rng.normal(0, 0.25))
    X = np.array(X)
    y = np.clip(np.array(y), 0.0, None)

    out = train_and_evaluate(X, y)
    results = out["results"]

    print(f"\nSynthetic dataset: {N} compounds, {X.shape[1]} descriptors")
    print(f"{'Model':<20}{'R^2':>8}{'RMSE (eV)':>12}")
    for name, m in results.items():
        print(f"{name:<20}{m['R2']:>8.3f}{m['RMSE']:>12.3f}")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name in results:
        yp = out["predictions"][name]
        yt = out["y_test"]
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(yt, yp, s=22, facecolors="none", edgecolors="tab:purple", linewidths=1.1)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Synthetic 'true' gap (eV)")
        ax.set_ylabel("ML predicted gap (eV)")
        ax.set_title(f"DEMO -- {name}")
        ax.text(0.05, 0.92, f"$R^2$={results[name]['R2']:.2f}\nRMSE={results[name]['RMSE']:.3f} eV",
                transform=ax.transAxes, va="top", fontsize=10)
        fig.text(0.5, 0.01, "SYNTHETIC DEMO DATA - NOT A RESULT", ha="center",
                 color="red", fontsize=8)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(f"figures/demo_parity_{name}.png", dpi=160)
        plt.close(fig)

    with open("results/demo_metrics.json", "w") as fh:
        json.dump({"WARNING": "SYNTHETIC DEMO DATA - NOT A RESULT",
                   "models": results}, fh, indent=2)

    print("\nDemo figures -> figures/demo_parity_*.png")
    print("Reminder: replace with real Materials Project data before reporting.")


if __name__ == "__main__":
    main()
