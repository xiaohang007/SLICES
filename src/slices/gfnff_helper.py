# -*- coding: utf-8 -*-
"""Topology-only GFN-FF helper used by SLICES reconstruction.

``SLICES.get_inner_p_target`` needs GFN-FF bond and angle parameters
(``blist``, ``vbond``, ``alist``, ``vangl``) for the finite neighbor table that
``SLICES.get_nbf_blist`` writes as ``testBonds_cut.top``. Two backends produce
the same ``gfnff_lists.json`` payload:

* ``xtb``: the legacy checked-in modified xtb binary (``XTB_MOD_PATH``).
* ``vendor``: the open-source GFN-FF library (pprcht/gfnff, LGPL-3.0-or-later)
  with the SLICES topology-export patch, bundled under ``slices/_vendor``. By
  default it uses the legacy 86-element xtb parameter table, so elements
  Z=1..86 follow the legacy helper; Z=87..103 use the library's own parameters.

Select with ``SLICES_GFNFF_HELPER_BACKEND`` (``vendor``, the default, or ``xtb``).
``SLICES_VENDOR_GFNFF_PARAM_MODE=modern`` switches the vendor backend to the
library's own parameters for all elements.

Why ``vendor`` is the default (MatterGPT MP-20 set, 2383 structures, CPU decode):
the reconstruction targets are identical for 88% of structures. Where they differ,
decoding with ``vendor`` matched the reference structure for 165/194 vs 79/194 with
xtb and never did worse with the standard matcher: xtb returns non-finite targets
for actinides and crashes outright on 39 inputs; lanthanides and the remaining
structures match the reference equally often with either backend.

Self-test: ``python -m slices.gfnff_helper``
"""
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

BACKEND_XTB = "xtb"
BACKEND_VENDOR = "vendor"
DEFAULT_BACKEND = BACKEND_VENDOR

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
LEGACY86_PARAM_FILE = _VENDOR_DIR / "gfnff_extra" / "param_gfnff_legacy86.xtb"

_BACKEND_ALIASES = {
    "": DEFAULT_BACKEND,
    "xtb": BACKEND_XTB, "legacy": BACKEND_XTB, "legacy_xtb": BACKEND_XTB,
    "vendor": BACKEND_VENDOR, "gfnff": BACKEND_VENDOR, "vendor_topology": BACKEND_VENDOR,
}
_PARAM_ALIASES = {
    "": "legacy86_overlay", "default": "legacy86_overlay",
    "legacy86": "legacy86_overlay", "legacy86_overlay": "legacy86_overlay",
    "modern": "modern", "internal": "modern", "none": "modern",
}
_SYMBOL_TO_Z = None


def selected_backend():
    name = os.environ.get("SLICES_GFNFF_HELPER_BACKEND", "").strip().lower()
    if name not in _BACKEND_ALIASES:
        raise ValueError(f"SLICES_GFNFF_HELPER_BACKEND must be 'xtb' or 'vendor', got {name!r}")
    return _BACKEND_ALIASES[name]


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
    """Parse ``testBonds_cut.top`` (nat, symbols, 20 neighbor rows) into (Z array, (20, nat) table)."""
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


def run_vendor(nbf_text):
    """Run the bundled GFN-FF library on an NBF table; returns the gfnff_lists.json payload."""
    from slices._vendor import gfnff
    numbers, nbf = parse_nbf_text(nbf_text)
    param_mode, param_file = _selected_params()
    with _env_default("SLICES_VENDOR_GFNFF_RAB_MODE", param_mode):
        data = gfnff.GFNFFTopology.from_nbf(numbers, nbf, printlevel=0, parametrisation=param_file).to_legacy_lists()
    data.setdefault("metadata", {}).update({"backend": BACKEND_VENDOR, "param_mode": param_mode})
    return data


def run_xtb(work_dir):
    """Run the legacy xtb helper on ``work_dir/testBonds_cut.top``; returns the gfnff_lists.json payload."""
    subprocess.call(os.environ["XTB_MOD_PATH"] + " --gfnff testBonds_cut.top --wrtopo blist,vbond,alist,vangl",
                    cwd=work_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(os.path.join(work_dir, "gfnff_lists.json"), "r") as f:
        txt = f.read().replace("********", "       0")  # xtb prints overflowing fields as asterisks
    return json.loads(txt)


def run_gfnff_helper(nbf_text, work_dir, backend=None):
    """Write ``testBonds_cut.top`` into ``work_dir``, run the selected backend, and return
    {blist, vbond, alist, vangl, ...}. ``work_dir/gfnff_lists.json`` holds the payload afterwards
    for both backends, so ``check_results`` debugging copies keep working."""
    backend = backend or selected_backend()
    with open(os.path.join(work_dir, "testBonds_cut.top"), "w") as f:
        f.write(nbf_text)
    if backend == BACKEND_XTB:
        return run_xtb(work_dir)
    data = run_vendor(nbf_text)
    with open(os.path.join(work_dir, "gfnff_lists.json"), "w") as f:
        json.dump(data, f)
    return data


def _self_test():
    """Methane topology plus a real SLICES (Na3MnCoNiO6) through both backends."""
    methane = "5\nC H H H H\n2 1 1 1 1\n3 0 0 0 0\n4 0 0 0 0\n5 0 0 0 0\n" + "0 0 0 0 0\n" * 15 + "4 1 1 1 1\n"
    data = run_vendor(methane)
    assert sorted(map(tuple, data["blist"])) == [(2, 1), (3, 1), (4, 1), (5, 1)], data["blist"]
    assert len(data["alist"]) == 6, len(data["alist"])
    print(f"methane: 4 bonds, 6 angles ({data['metadata']['param_mode']})")

    from slices.core import SLICES
    example = ('o x a DDO j OOO o Na Na Na Mn Co Ni O O O O O O 0 8 oo- 0 10 o-o 0 10 -oo 0 6 o-- 0 6 -o- 0 11 '
               'ooo 1 6 ooo 1 11 +o+ 1 11 o++ 1 7 +oo 1 7 o+o 1 9 +++ 2 7 ooo 2 9 +o+ 2 9 o++ 2 8 +oo 2 8 o+o 2 '
               '10 oo+ 3 11 ooo 3 6 o-o 3 6 -oo 3 9 +oo 3 9 o+o 3 7 ooo 4 10 ooo 4 8 +oo 4 8 o+o 4 11 +oo 4 11 '
               'o+o 4 6 ooo 5 9 ooo 5 7 o-o 5 7 -oo 5 10 o-o 5 10 -oo 5 8 ooo')
    s = SLICES()
    s.from_SLICES(example)
    results = {}
    for backend in (BACKEND_VENDOR, BACKEND_XTB):
        os.environ["SLICES_GFNFF_HELPER_BACKEND"] = backend
        try:
            results[backend] = s.get_inner_p_target()
        finally:
            os.environ.pop("SLICES_GFNFF_HELPER_BACKEND", None)
    (ipv, civ, cwv), (ipx, cix, cwx) = results[BACKEND_VENDOR], results[BACKEND_XTB]
    assert civ == cix and np.allclose(ipv, ipx, atol=1e-8) and np.allclose(cwv, cwx, atol=1e-8)
    print(f"Na3MnCoNiO6: vendor and xtb targets agree "
          f"(max |d inner_p| = {np.abs(np.asarray(ipv) - np.asarray(ipx)).max():.1e})")


if __name__ == "__main__":
    _self_test()
