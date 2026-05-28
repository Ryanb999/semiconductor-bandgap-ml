"""
fetch_data.py
-------------
Download a semiconductor dataset from the Materials Project using the current
`mp-api` client, and save it to data/semiconductors.csv.

This is the step that produces YOUR real data. It requires:
  1. `pip install mp-api`  (see requirements.txt)
  2. A free Materials Project API key from https://next-gen.materialsproject.org/api
     Set it as an environment variable:  export MP_API_KEY="your_key_here"
     (or pass --api-key on the command line).

Usage
-----
    python src/fetch_data.py                       # default: stable semiconductors, gap 0.1-6 eV
    python src/fetch_data.py --max-gap 4 --max-ehull 0.05
    python src/fetch_data.py --elements Si O        # restrict to a chemistry

The query mirrors the descriptor-driven philosophy of Linton & Aidhy: we keep
composition + the target property (band gap) and a few auxiliary thermodynamic
fields that are cheap to query and useful as sanity checks.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Fetch a semiconductor dataset from Materials Project.")
    p.add_argument("--api-key", default=os.environ.get("MP_API_KEY"),
                   help="Materials Project API key (or set MP_API_KEY env var).")
    p.add_argument("--min-gap", type=float, default=0.1,
                   help="Minimum band gap in eV (exclude metals). Default 0.1.")
    p.add_argument("--max-gap", type=float, default=6.0,
                   help="Maximum band gap in eV (exclude wide-gap insulators). Default 6.0.")
    p.add_argument("--max-ehull", type=float, default=0.10,
                   help="Maximum energy above hull in eV/atom (stability filter). Default 0.10.")
    p.add_argument("--max-elements", type=int, default=4,
                   help="Maximum number of distinct elements per compound. Default 4.")
    p.add_argument("--elements", nargs="*", default=None,
                   help="Optional: restrict to compounds containing ONLY these elements.")
    p.add_argument("--out", default="data/semiconductors.csv",
                   help="Output CSV path. Default data/semiconductors.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        sys.exit("ERROR: No API key. Set MP_API_KEY or pass --api-key. "
                 "Get one free at https://next-gen.materialsproject.org/api")

    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("ERROR: mp-api is not installed. Run: pip install mp-api")

    fields = [
        "material_id", "formula_pretty", "band_gap", "is_gap_direct",
        "formation_energy_per_atom", "energy_above_hull", "nelements",
        "nsites", "density", "symmetry",
    ]

    print("Querying Materials Project ...")
    with MPRester(args.api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(args.min_gap, args.max_gap),
            energy_above_hull=(0.0, args.max_ehull),
            is_metal=False,
            num_elements=(1, args.max_elements),
            elements=args.elements,         # None => no restriction
            fields=fields,
        )

    rows = []
    for d in docs:
        rows.append({
            "material_id": str(d.material_id),
            "formula_pretty": d.formula_pretty,
            "band_gap": d.band_gap,
            "is_gap_direct": d.is_gap_direct,
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": d.energy_above_hull,
            "nelements": d.nelements,
            "nsites": d.nsites,
            "density": d.density,
            "crystal_system": (d.symmetry.crystal_system.value
                               if getattr(d, "symmetry", None) else None),
        })

    df = pd.DataFrame(rows)
    # Drop anything with a missing target and de-duplicate on material_id.
    df = df.dropna(subset=["band_gap", "formula_pretty"]).drop_duplicates("material_id")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} semiconductor entries -> {args.out}")
    print(df[["formula_pretty", "band_gap"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
