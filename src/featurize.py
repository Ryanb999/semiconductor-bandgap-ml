"""
featurize.py
------------
Turn chemical formulas into composition-based descriptors for ML.

Two backends are provided:

  1. PREFERRED -- matminer's Magpie ElementProperty preset (132 features:
     composition-weighted statistics of elemental properties). This is the
     community-standard featurization and what you should report in the paper.
         pip install matminer pymatgen

  2. FALLBACK -- a small, dependency-free elemental featurizer over a built-in
     property table. It is automatically used if matminer is not installed, so
     the pipeline always runs. It produces fewer features but follows the same
     idea (composition-weighted mean / range / std of elemental properties),
     which is exactly the descriptor philosophy of Linton & Aidhy (bond/element
     chemistry -> property).

Input : a CSV with a `formula_pretty` column and a target column (band_gap).
Output: data/features.csv  (descriptors + target, ready for train_models.py)

Usage
-----
    python src/featurize.py --in data/semiconductors.csv --target band_gap
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Fallback elemental property table (subset; extend as needed).
# Columns: Z, atomic_mass, electronegativity(Pauling), atomic_radius(pm),
#          valence_electrons, group, period
# --------------------------------------------------------------------------- #
ELEM = {
    "H":  (1, 1.008, 2.20, 25, 1, 1, 1),
    "Li": (3, 6.94, 0.98, 145, 1, 1, 2),
    "Be": (4, 9.012, 1.57, 105, 2, 2, 2),
    "B":  (5, 10.81, 2.04, 85, 3, 13, 2),
    "C":  (6, 12.011, 2.55, 70, 4, 14, 2),
    "N":  (7, 14.007, 3.04, 65, 5, 15, 2),
    "O":  (8, 15.999, 3.44, 60, 6, 16, 2),
    "F":  (9, 18.998, 3.98, 50, 7, 17, 2),
    "Na": (11, 22.99, 0.93, 180, 1, 1, 3),
    "Mg": (12, 24.305, 1.31, 150, 2, 2, 3),
    "Al": (13, 26.982, 1.61, 125, 3, 13, 3),
    "Si": (14, 28.085, 1.90, 110, 4, 14, 3),
    "P":  (15, 30.974, 2.19, 100, 5, 15, 3),
    "S":  (16, 32.06, 2.58, 100, 6, 16, 3),
    "Cl": (17, 35.45, 3.16, 100, 7, 17, 3),
    "K":  (19, 39.098, 0.82, 220, 1, 1, 4),
    "Ca": (20, 40.078, 1.00, 180, 2, 2, 4),
    "Sc": (21, 44.956, 1.36, 160, 3, 3, 4),
    "Ti": (22, 47.867, 1.54, 140, 4, 4, 4),
    "V":  (23, 50.942, 1.63, 135, 5, 5, 4),
    "Cr": (24, 51.996, 1.66, 140, 6, 6, 4),
    "Mn": (25, 54.938, 1.55, 140, 7, 7, 4),
    "Fe": (26, 55.845, 1.83, 140, 8, 8, 4),
    "Co": (27, 58.933, 1.88, 135, 9, 9, 4),
    "Ni": (28, 58.693, 1.91, 135, 10, 10, 4),
    "Cu": (29, 63.546, 1.90, 135, 11, 11, 4),
    "Zn": (30, 65.38, 1.65, 135, 12, 12, 4),
    "Ga": (31, 69.723, 1.81, 130, 3, 13, 4),
    "Ge": (32, 72.63, 2.01, 125, 4, 14, 4),
    "As": (33, 74.922, 2.18, 115, 5, 15, 4),
    "Se": (34, 78.971, 2.55, 115, 6, 16, 4),
    "Br": (35, 79.904, 2.96, 115, 7, 17, 4),
    "Rb": (37, 85.468, 0.82, 235, 1, 1, 5),
    "Sr": (38, 87.62, 0.95, 200, 2, 2, 5),
    "Y":  (39, 88.906, 1.22, 180, 3, 3, 5),
    "Zr": (40, 91.224, 1.33, 155, 4, 4, 5),
    "Nb": (41, 92.906, 1.60, 145, 5, 5, 5),
    "Mo": (42, 95.95, 2.16, 145, 6, 6, 5),
    "Ag": (47, 107.868, 1.93, 160, 11, 11, 5),
    "Cd": (48, 112.414, 1.69, 155, 12, 12, 5),
    "In": (49, 114.818, 1.78, 155, 3, 13, 5),
    "Sn": (50, 118.71, 1.96, 145, 4, 14, 5),
    "Sb": (51, 121.76, 2.05, 145, 5, 15, 5),
    "Te": (52, 127.6, 2.10, 140, 6, 16, 5),
    "I":  (53, 126.904, 2.66, 140, 7, 17, 5),
    "Cs": (55, 132.905, 0.79, 260, 1, 1, 6),
    "Ba": (56, 137.327, 0.89, 215, 2, 2, 6),
    "La": (57, 138.905, 1.10, 195, 3, 3, 6),
    "Hf": (72, 178.49, 1.30, 155, 4, 4, 6),
    "Ta": (73, 180.948, 1.50, 145, 5, 5, 6),
    "W":  (74, 183.84, 2.36, 135, 6, 6, 6),
    "Au": (79, 196.967, 2.54, 135, 11, 11, 6),
    "Hg": (80, 200.592, 2.00, 150, 12, 12, 6),
    "Tl": (81, 204.38, 1.62, 190, 3, 13, 6),
    "Pb": (82, 207.2, 2.33, 180, 4, 14, 6),
    "Bi": (83, 208.98, 2.02, 160, 5, 15, 6),
    "Pd": (46, 106.42, 2.20, 140, 10, 10, 5),
    "Pt": (78, 195.084, 2.28, 135, 10, 10, 6),
}
PROP_NAMES = ["Z", "mass", "electroneg", "radius", "valence", "group", "period"]

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict:
    """Very small formula parser, e.g. 'GaAs' -> {Ga:1, As:1}, 'SiO2' -> {Si:1,O:2}.
    Handles simple fractional/integer subscripts. No nested parentheses."""
    counts: dict[str, float] = {}
    for sym, num in _FORMULA_RE.findall(formula):
        if not sym:
            continue
        n = float(num) if num else 1.0
        counts[sym] = counts.get(sym, 0.0) + n
    return counts


def fallback_features(formulas: pd.Series) -> pd.DataFrame:
    """Composition-weighted mean, range, and std of elemental properties."""
    rows = []
    skipped = 0
    for f in formulas:
        comp = parse_formula(str(f))
        elems = [e for e in comp if e in ELEM]
        if not elems:
            rows.append(None)
            skipped += 1
            continue
        total = sum(comp[e] for e in elems)
        weights = np.array([comp[e] / total for e in elems])
        mat = np.array([ELEM[e] for e in elems], dtype=float)  # (n_elems, n_props)

        feat = {}
        for j, pname in enumerate(PROP_NAMES):
            col = mat[:, j]
            feat[f"{pname}_mean"] = float(np.dot(weights, col))
            feat[f"{pname}_range"] = float(col.max() - col.min())
            feat[f"{pname}_std"] = float(np.sqrt(np.dot(weights, (col - np.dot(weights, col)) ** 2)))
        feat["n_elements"] = len(elems)
        rows.append(feat)

    if skipped:
        print(f"  NOTE: {skipped} formula(s) had elements outside the fallback table "
              f"and were dropped. Install matminer for full element coverage.")
    return pd.DataFrame([r for r in rows if r is not None],
                        index=[i for i, r in enumerate(rows) if r is not None])


def magpie_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full Magpie featurization via matminer (preferred)."""
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition

    df = StrToComposition().featurize_dataframe(df, "formula_pretty", ignore_errors=True)
    ep = ElementProperty.from_preset("magpie")
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True)
    feat_cols = ep.feature_labels()
    return df[feat_cols]


def parse_args():
    p = argparse.ArgumentParser(description="Featurize formulas into ML descriptors.")
    p.add_argument("--in", dest="infile", default="data/semiconductors.csv")
    p.add_argument("--target", default="band_gap")
    p.add_argument("--out", default="data/features.csv")
    p.add_argument("--force-fallback", action="store_true",
                   help="Use the built-in featurizer even if matminer is installed.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.infile)
    if "formula_pretty" not in df.columns or args.target not in df.columns:
        sys.exit(f"ERROR: input must contain 'formula_pretty' and '{args.target}' columns.")

    use_magpie = not args.force_fallback
    if use_magpie:
        try:
            feats = magpie_features(df.copy())
            backend = "matminer/Magpie"
        except ImportError:
            use_magpie = False
    if not use_magpie:
        feats = fallback_features(df["formula_pretty"])
        backend = "built-in fallback"

    feats = feats.copy()
    feats[args.target] = df.loc[feats.index, args.target].values
    feats = feats.dropna()
    feats.to_csv(args.out, index=False)

    print(f"Featurizer backend : {backend}")
    print(f"Feature matrix      : {feats.shape[0]} rows x {feats.shape[1]-1} descriptors")
    print(f"Saved               -> {args.out}")


if __name__ == "__main__":
    main()
