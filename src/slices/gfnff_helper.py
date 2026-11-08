# -*- coding: utf-8 -*-
"""Topology-only GFN-FF helper used by SLICES reconstruction.

``SLICES.get_inner_p_target`` needs GFN-FF bond and angle parameters
(``blist``, ``vbond``, ``alist``, ``vangl``) for the finite neighbor table that
``SLICES.get_nbf_blist`` builds (the ``testBonds_cut.top`` text format). They
come from the open-source GFN-FF library (pprcht/gfnff, LGPL-3.0-or-later) with
the SLICES topology-export patch, bundled under ``slices/_vendor`` (portable
manylinux2014 x86_64 build, loaded with ctypes, independent of the Python version).

By default it uses the 86-element parameter table of the modified xtb helper that
SLICES used before, so elements Z=1..86 follow that parametrisation and Z=87..103
use the library's own parameters. ``SLICES_VENDOR_GFNFF_PARAM_MODE=modern`` uses
the library's own parameters for all elements; ``SLICES_VENDOR_GFNFF_PARAM_FILE``
points to another xtb-style parameter table.

This replaced the checked-in modified xtb binary. On the 2383 MatterGPT MP-20
structures the reconstruction targets were identical for 88%; where they differed,
decoding matched the reference structure for 165/194 vs 79/194 with xtb (xtb gave
non-finite targets for actinides and crashed on 39 inputs). A 40k-structure
benchmark of the same library in the symSLICES code showed the same picture
(+262 net strict matches, -30 net loose matches out of 40000).

Self-test: ``python -m slices.gfnff_helper``
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
LEGACY86_PARAM_FILE = _VENDOR_DIR / "gfnff_extra" / "param_gfnff_legacy86.xtb"

_PARAM_ALIASES = {
    "": "legacy86_overlay", "default": "legacy86_overlay",
    "legacy86": "legacy86_overlay", "legacy86_overlay": "legacy86_overlay",
    "modern": "modern", "internal": "modern", "none": "modern",
}
_SYMBOL_TO_Z = None


def _selected_params():
    """Return (param_mode, parameter file path or '' for the library's own parameters)."""
    mode = os.environ.get("SLICES_VENDOR_GFNFF_PARAM_MODE", "").strip().lower()
    if mode not in _PARAM_ALIASES:
        raise ValueError(f"SLICES_VENDOR_GFNFF_PARAM_MODE must be 'legacy86_overlay' or 'modern', got {mode!r}")
    mode = _PARAM_ALIASES[mode]
    if mode == "modern":
        return mode, ""
    path = Path(os.environ.get("SLICES_VENDOR_GFNFF_PARAM_FILE", "") or LEGACY86_PARAM_FILE).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GFN-FF legacy86 parameter file not found: {path}")
    return mode, str(path)


@contextmanager
def _env_default(key, value):
    """Set ``key`` only if unset (the patched library reads it), restoring afterwards."""
    previous = os.environ.get(key)
    os.environ.setdefault(key, value)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def parse_nbf_text(nbf_text):
    """Parse ``testBonds_cut.top`` text (nat, symbols, 20 neighbor rows) into (Z array, (20, nat) table)."""
    global _SYMBOL_TO_Z
    if _SYMBOL_TO_Z is None:
        from pymatgen.core.periodic_table import Element
        _SYMBOL_TO_Z = {el.symbol: el.Z for el in Element}
    lines = [line for line in str(nbf_text).splitlines() if line.strip()]
    if len(lines) < 22:
        raise ValueError(f"NBF text needs nat, symbols and 20 neighbor rows; got {len(lines)} lines")
    nat = int(lines[0].split()[0])
    symbols = lines[1].split()
    if len(symbols) != nat:
        raise ValueError(f"NBF symbol row has {len(symbols)} entries, expected {nat}")
    nbf = np.array([[int(v) for v in line.split()] for line in lines[2:22]], dtype=np.int32)
    if nbf.shape != (20, nat):
        raise ValueError(f"NBF neighbor table has shape {nbf.shape}, expected (20, {nat})")
    return np.array([_SYMBOL_TO_Z[s] for s in symbols], dtype=np.int32), nbf


def run_gfnff_helper(nbf_text):
    """Run GFN-FF on an NBF table; returns {blist, vbond, alist, vangl, metadata} (1-based atom indices)."""
    from slices._vendor import gfnff
    numbers, nbf = parse_nbf_text(nbf_text)
    param_mode, param_file = _selected_params()
    with _env_default("SLICES_VENDOR_GFNFF_RAB_MODE", param_mode):
        data = gfnff.GFNFFTopology.from_nbf(numbers, nbf, printlevel=0, parametrisation=param_file).to_legacy_lists()
    data.setdefault("metadata", {}).update({"param_mode": param_mode})
    return data


def write_debug_files(nbf_text, data, directory="."):
    """Write the helper input and output (``testBonds_cut.top``, ``gfnff_lists.json``) for ``check_results``."""
    with open(os.path.join(directory, "testBonds_cut.top"), "w") as f:
        f.write(nbf_text)
    with open(os.path.join(directory, "gfnff_lists.json"), "w") as f:
        json.dump(data, f)


def _self_test():
    """Methane topology plus a real SLICES (Na3MnCoNiO6) against stored reference targets."""
    methane = "5\nC H H H H\n2 1 1 1 1\n3 0 0 0 0\n4 0 0 0 0\n5 0 0 0 0\n" + "0 0 0 0 0\n" * 15 + "4 1 1 1 1\n"
    data = run_gfnff_helper(methane)
    assert sorted(map(tuple, data["blist"])) == [(2, 1), (3, 1), (4, 1), (5, 1)], data["blist"]
    assert len(data["alist"]) == 6, len(data["alist"])
    print(f"methane: 4 bonds, 6 angles ({data['metadata']['param_mode']})")

    from slices.core import SLICES
    # First structure of MatterGPT/0_dataset/mp20_eform_bandgap_nonmetal.csv
    example = ('o x a DDO j OOO o Na Na Na Mn Co Ni O O O O O O 0 8 oo- 0 10 o-o 0 10 -oo 0 6 o-- 0 6 -o- 0 11 '
               'ooo 1 6 ooo 1 11 +o+ 1 11 o++ 1 7 +oo 1 7 o+o 1 9 +++ 2 7 ooo 2 9 +o+ 2 9 o++ 2 8 +oo 2 8 o+o 2 '
               '10 oo+ 3 11 ooo 3 6 o-o 3 6 -oo 3 9 +oo 3 9 o+o 3 7 ooo 4 10 ooo 4 8 +oo 4 8 o+o 4 11 +oo 4 11 '
               'o+o 4 6 ooo 5 9 ooo 5 7 o-o 5 7 -oo 5 10 o-o 5 10 -oo 5 8 ooo')
    s = SLICES()
    s.from_SLICES(example)
    inner_p, inds, weights = s.get_inner_p_target()
    got = (float(np.trace(inner_p)), float(np.abs(inner_p).sum()), len(inds[0]), float(sum(weights)))
    want = (178.74144, 178.74144, 126, 61.23)  # computed with the xtb-era code path (commit d40c909)
    assert np.allclose(got, want, rtol=1e-6), (got, want)
    print(f"Na3MnCoNiO6: reconstruction targets match the stored reference {want}")


if __name__ == "__main__":
    _self_test()
