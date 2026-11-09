# -*- coding: utf-8 -*-
"""Compare the force fields used in the last step of SLICES decoding on this machine.

Decodes real SLICES from the MatterGPT dataset with each relax_model, and reports the decode
time per structure, failures, and the formation energy error against the DFT values in the
dataset (computed like MatterGPT/2_decode, using chemPotMP.json).

Run from the repository root, e.g.:
    python -m slices.relax_backends --prepare          # compile/cache the NequIP models first
    python examples/3.benchmark_relax_models.py --n 32
    python examples/3.benchmark_relax_models.py --n 32 --models auto nequip-m+l
    SLICES_DEVICE=cuda python examples/3.benchmark_relax_models.py --models nequip-l   # GPU (opt-in)

The timing is for one process; MatterGPT decoding runs several such processes in parallel.
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slices.core import SLICES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def formation_energy(structure, energy_per_atom, chem_pot):
    try:
        prim = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    except Exception:
        prim = structure
    comp = prim.composition
    total = energy_per_atom * comp.num_atoms - sum(amt * chem_pot[el] for el, amt in comp.get_el_amt_dict().items())
    return total / comp.num_atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=os.path.join(ROOT, "MatterGPT/0_dataset/mp20_eform_bandgap_nonmetal.csv"))
    parser.add_argument("--chem-pot", default=os.path.join(ROOT, "MatterGPT/2_decode/workflow/chemPotMP.json"))
    parser.add_argument("--n", type=int, default=32, help="number of SLICES to decode (default 32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-atoms", type=int, default=20,
                        help="skip SLICES with more atoms (default 20, the fast relax() path)")
    parser.add_argument("--models", nargs="+", default=["auto", "nequip-m+l"])
    parser.add_argument("--out", default=None, help="optional JSON file for per-structure results")
    args = parser.parse_args()

    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    chem_pot = json.load(open(args.chem_pot, encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    probe = SLICES(relax_model=None)
    picked = []
    for r in rows:
        try:
            probe.from_SLICES(r["SLICES"])
        except Exception:
            continue
        if len(probe.atom_types) <= args.max_atoms:
            picked.append(r)
        if len(picked) == args.n:
            break
    print(f"{len(picked)} SLICES from {os.path.relpath(args.csv, ROOT)} (seed {args.seed}, <= {args.max_atoms} atoms)\n")

    report, details = [], {}
    for model in args.models:
        t0 = time.time()
        try:
            cg = SLICES(relax_model=model)
        except Exception as exc:
            print(f"{model}: cannot be used here ({type(exc).__name__}: {exc})\n")
            continue
        setup = time.time() - t0
        desc = getattr(cg.relaxer, "description", "")
        device = getattr(cg.relaxer, "device", "cpu")
        try:
            cg.SLICES2structure(picked[0]["SLICES"])  # warm-up
        except Exception:
            pass
        times, errors, failed = [], [], 0
        details[model] = []
        for r in picked:
            t1 = time.time()
            try:
                structure, e = cg.SLICES2structure(r["SLICES"])
                ef = formation_energy(structure, e, chem_pot)
                err = ef - float(r["formation_energy_per_atom"])
                errors.append(abs(err))
                details[model].append({"slices": r["SLICES"], "eform": ef, "ref": float(r["formation_energy_per_atom"])})
            except Exception as exc:
                failed += 1
                details[model].append({"slices": r["SLICES"], "error": repr(exc)[:200]})
            times.append(time.time() - t1)
        line = (f"{model:11s} -> {cg.relax_model:10s} on {device:4s}  setup {setup:5.1f}s  "
                f"decode {statistics.mean(times):6.2f} s/structure  failed {failed}/{len(picked)}")
        if errors:
            line += (f"  |eform - DFT| mean {statistics.mean(errors):.3f} median {statistics.median(errors):.3f} "
                     f"eV/atom, <=0.05 eV: {sum(e <= 0.05 for e in errors)}/{len(errors)}")
        print(line + (f"\n            {desc}" if desc else "") + "\n", flush=True)
        report.append(line)

    print("summary:\n  " + "\n  ".join(report))
    if args.out:
        json.dump(details, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
