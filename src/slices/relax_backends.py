# -*- coding: utf-8 -*-
"""Force fields for the final relaxation step of SLICES decoding.

``SLICES(relax_model=...)`` accepts:

* ``"auto"`` (default): ``"nequip-s+l"`` on every machine.
* ``"nequip-s+l"`` / ``"nequip-m+l"``: relax with NequIP-OAM-S / -M, energy from NequIP-OAM-L.
* ``"nequip-l"``: relax with NequIP-OAM-L (practical only on a GPU).
* ``None`` or ``"none"``: no relaxer.

All NequIP variants report the NequIP-OAM-L energy. Models run on the CPU unless
``SLICES_DEVICE=cuda``. On the 2383 MP-20 structures (CPU), ``nequip-s+l`` reproduced 93.8% of
the reference structures (M3GNet pipeline: 88.7%); ``nequip-m+l`` reached 94.8% at twice the
CPU time. A GPU made decoding only ~12% faster, since most of the decode time is the CPU-side
GFN-FF and lattice optimization steps, so the GPU path is opt-in.

Environment variables:

* ``SLICES_RELAX_MODEL``: used when ``relax_model="auto"``, e.g. ``nequip-m+l``.
* ``SLICES_DEVICE``: ``cpu`` (default) or ``cuda``. GPU runs compile the models with
  AOTInductor, which needs the CUDA toolkit (``CUDA_HOME``); without it they fall back to
  TorchScript, which on a GPU is slower than the CPU path. ``uv sync --extra oeq`` adds
  OpenEquivariance kernels for the GPU.
* ``SLICES_MODEL_CACHE``: directory for compiled NequIP models (default ``~/.cache/slices/nequip``).
* ``SLICES_TORCH_THREADS``: PyTorch CPU threads per process (default ``OMP_NUM_THREADS`` or 1;
  one thread per process with many processes gives the best throughput for small cells).
* ``SLICES_NEQUIP_TF32``: ``1`` compiles GPU models with TF32 (faster on Ampere and newer GPUs,
  slightly less precise).
* ``SLICES_NEQUIP_MODEL_S`` / ``_M`` / ``_L``: a compiled model file (``.nequip.pth`` or
  ``.nequip.pt2``) to use as is, or a local ``.nequip.zip`` package to compile instead of
  downloading from nequip.net.

Compiled models are tied to the GPU, CUDA, torch and nequip versions, so they are compiled on
first use on each machine and cached. Run ``python -m slices.relax_backends --prepare`` once
before starting many parallel workers.
"""
import importlib.util
import os
import re
import subprocess
import sys
import time
import types

NEQUIP_MODEL_IDS = {
    "S": "mir-group/NequIP-OAM-S:0.1",
    "M": "mir-group/NequIP-OAM-M:0.1",
    "L": "mir-group/NequIP-OAM-L:0.1",
}
# relax_model -> (model used for relaxation, model used for the final energy or None)
NEQUIP_PLANS = {
    "nequip-l": ("L", None),
    "nequip-s+l": ("S", "L"),
    "nequip-m+l": ("M", "L"),
}
RELAX_MODELS = ("auto", "none") + tuple(NEQUIP_PLANS)
OPTIMIZERS = ("BFGS", "LBFGS", "FIRE", "FIRE+BFGS")
FIRE_PRESTEPS = 10  # FIRE steps before switching to BFGS in "FIRE+BFGS"

_CALCULATORS = {}
_ANNOUNCED = set()


class RelaxBackendError(RuntimeError):
    """A NequIP model could not be compiled, loaded or evaluated."""


def _log(msg):
    print(f"[SLICES relax] {msg}", file=sys.stderr, flush=True)


def _announce(msg):
    if msg not in _ANNOUNCED:
        _ANNOUNCED.add(msg)
        _log(msg)


def nequip_available():
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("nequip") is not None


def _nequip_install_hint():
    return "set up the environment with `uv sync --extra cpu` (or cu126 / cu128), see README"


def choose_device():
    """``SLICES_DEVICE`` (``cpu`` or ``cuda``), default ``cpu``."""
    device = os.environ.get("SLICES_DEVICE", "").strip().lower() or "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"SLICES_DEVICE must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            _announce("SLICES_DEVICE=cuda but torch sees no CUDA GPU; using the CPU")
            return "cpu"
    return device


def resolve_relax_model(relax_model):
    """Return ``(name, device)`` for a ``relax_model`` value (device None without a relaxer)."""
    if relax_model is None:
        return "none", None
    name = str(relax_model).strip().lower()
    if name == "auto":
        name = os.environ.get("SLICES_RELAX_MODEL", "auto").strip().lower() or "auto"
    if name == "m3gnet":
        raise ValueError("M3GNet support has been removed; use relax_model='auto' or one of "
                         f"{tuple(NEQUIP_PLANS)}")
    if name not in RELAX_MODELS:
        raise ValueError(f"Unknown relax_model {relax_model!r}; choose from {RELAX_MODELS}")
    if name == "auto":
        name = "nequip-s+l"
    if name in NEQUIP_PLANS:
        if not nequip_available():
            raise ImportError(f"relax_model={name!r} needs torch and nequip ({_nequip_install_hint()})")
        return name, choose_device()
    return name, None


def make_relaxer(relax_model="auto", optimizer="BFGS"):
    """Build the relaxer for ``relax_model``. Returns ``(relaxer or None, resolved name)``."""
    name, device = resolve_relax_model(relax_model)
    if name == "none":
        return None, "none"
    try:
        relaxer = NequIPRelaxer(name, device, optimizer)
    except Exception as exc:
        if device != "cuda":
            raise
        _announce(f"GPU setup failed ({type(exc).__name__}: {exc}); falling back to CPU nequip-s+l")
        name, device = "nequip-s+l", "cpu"
        relaxer = NequIPRelaxer(name, device, optimizer)
    _announce(f"relax_model={name} on {device} ({relaxer.description})")
    return relaxer, name


# ----------------------------------------------------------------------------- NequIP

def _cache_dir():
    path = os.environ.get("SLICES_MODEL_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "slices", "nequip")
    os.makedirs(path, exist_ok=True)
    return path


def _tf32():
    return os.environ.get("SLICES_NEQUIP_TF32", "").strip() in ("1", "true", "yes")


def _compile_tiers(device):
    """Compilation attempts in order of preference: (mode, use OpenEquivariance)."""
    if device == "cpu":
        # On CPU, TorchScript was 3-4x faster than AOTInductor in our tests.
        return [("torchscript", False)]
    tiers = []
    if importlib.util.find_spec("openequivariance") is not None:
        tiers.append(("aotinductor", True))
    tiers += [("aotinductor", False), ("torchscript", False)]
    return tiers


def _compiled_name(size, device, mode, oeq):
    import torch
    import nequip

    parts = [f"NequIP-OAM-{size}-0.1", mode + ("-oeq" if oeq else ""), device]
    if device == "cuda":
        gpu = re.sub(r"[^A-Za-z0-9]+", "-", torch.cuda.get_device_name(0)).strip("-")
        major, minor = torch.cuda.get_device_capability(0)
        parts += [gpu, f"sm{major}{minor}", f"cu{torch.version.cuda}"] + (["tf32"] if _tf32() else [])
    parts += [f"torch{torch.__version__.split('+')[0]}", f"nequip{nequip.__version__}"]
    ext = ".nequip.pth" if mode == "torchscript" else ".nequip.pt2"
    return "__".join(parts) + ext


def _model_source(size):
    """Local package path for compilation (downloads into nequip's cache if needed)."""
    override = os.environ.get(f"SLICES_NEQUIP_MODEL_{size}")
    if override:
        return override
    model_ref = "nequip.net:" + NEQUIP_MODEL_IDS[size]
    try:  # resolve to the cached .nequip.zip so download problems surface here, not as compile failures
        from nequip.model.saved_models.load_utils import _get_model_file_path
        with _get_model_file_path(model_ref) as path:
            return str(path)
    except ImportError:
        return model_ref


def _compile(source, out_path, device, mode, oeq):
    tmp = os.path.join(os.path.dirname(out_path), f".tmp{os.getpid()}-{os.path.basename(out_path)}")
    cmd = [sys.executable, "-m", "nequip.scripts.compile", source, tmp, "--mode", mode, "--device", device]
    if mode == "aotinductor":
        # default shape bounds start at 2 atoms; primitive cells of elements can have 1
        cmd += ["--target", "ase", "--num-nodes", "1,inf"]
    if oeq:
        cmd += ["--modifiers", "enable_OpenEquivariance"]
    if device == "cuda" and _tf32():
        cmd += ["--tf32"]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RelaxBackendError(f"nequip-compile failed ({mode}{', OpenEquivariance' if oeq else ''}):\n"
                                + proc.stdout[-3000:])
    os.replace(tmp, out_path)
    _log(f"compiled {os.path.basename(out_path)} in {time.time() - t0:.0f} s")


def _self_test(calc):
    from ase.build import bulk
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()
    if not energy == energy:  # NaN
        raise RelaxBackendError("model returned NaN energy in self-test")


def _set_cpu_threads():
    import torch
    if getattr(_set_cpu_threads, "_done", False):
        return
    threads = int(os.environ.get("SLICES_TORCH_THREADS") or os.environ.get("OMP_NUM_THREADS") or 1)
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set before any inter-op parallel work
        pass
    _set_cpu_threads._done = True


def _load_calculator(path, device):
    from nequip.integrations.ase import NequIPCalculator
    if "aotinductor-oeq" in os.path.basename(path):
        import openequivariance  # noqa: F401  registers the libtorch_tp_jit ops the package calls
    # OAM type names are element symbols, so the identity map is correct (and silences nequip's warning)
    return NequIPCalculator.from_compiled_model(path, device=device, chemical_species_to_atom_type_map=True)


def get_calculator(size, device):
    """ASE calculator for NequIP-OAM-<size> on ``device``, compiled and cached on first use."""
    key = (size, device)
    if key in _CALCULATORS:
        return _CALCULATORS[key]
    if device == "cpu":
        _set_cpu_threads()
    override = os.environ.get(f"SLICES_NEQUIP_MODEL_{size}", "")
    if override.endswith((".nequip.pth", ".nequip.pt2")):
        calc = _load_calculator(override, device)
        _self_test(calc)
        _CALCULATORS[key] = (calc, os.path.basename(override))
        return _CALCULATORS[key]

    import fcntl

    cache = _cache_dir()
    errors = []
    for mode, oeq in _compile_tiers(device):
        path = os.path.join(cache, _compiled_name(size, device, mode, oeq))
        failed_marker = path + ".failed"
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # one process compiles, the others wait and reuse it
            if os.path.exists(failed_marker):
                errors.append(f"{os.path.basename(path)}: failed earlier, see {failed_marker}")
                continue
            if not os.path.exists(path):
                _log(f"compiling NequIP-OAM-{size} for {device} ({mode}{', OpenEquivariance' if oeq else ''}); "
                     f"first use on this machine can take several minutes")
                try:
                    _compile(_model_source(size), path, device, mode, oeq)
                except RelaxBackendError as exc:
                    with open(failed_marker, "w") as fh:
                        fh.write(str(exc))
                    errors.append(str(exc).splitlines()[0])
                    _log(f"{exc.args[0].splitlines()[0]} -> trying the next option "
                         f"(delete {failed_marker} to retry)")
                    continue
        try:
            calc = _load_calculator(path, device)
            _self_test(calc)
        except Exception as exc:
            errors.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
            _log(f"could not use {os.path.basename(path)} ({type(exc).__name__}: {exc}); trying the next option")
            continue
        _CALCULATORS[key] = (calc, os.path.basename(path))
        return _CALCULATORS[key]
    raise RelaxBackendError(f"no usable NequIP-OAM-{size} model for {device}:\n  " + "\n  ".join(errors))


def _cell_filter(atoms):
    try:
        from ase.filters import ExpCellFilter
    except ImportError:  # ase < 3.23
        from ase.constraints import ExpCellFilter
    return ExpCellFilter(atoms)


def _optimize(target, optimizer, fmax, steps):
    from ase.optimize import BFGS, FIRE, LBFGS

    classes = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}
    if optimizer == "FIRE+BFGS":
        plan = [(FIRE, min(FIRE_PRESTEPS, steps)), (BFGS, steps)]
    else:
        plan = [(classes[optimizer], steps)]
    used = 0
    for cls, nmax in plan:
        nmax = min(nmax, steps - used)
        if nmax <= 0:
            break
        opt = cls(target, logfile=None)
        converged = opt.run(fmax=fmax, steps=nmax)
        used += opt.nsteps
        if converged:
            break
    return used


class NequIPRelaxer:
    """``relax()`` returns ``final_structure`` and a ``trajectory`` whose ``energies[-1]`` is the
    total energy (eV) of the final structure (the interface used by ``SLICES.relax``)."""

    def __init__(self, name, device, optimizer="BFGS"):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer {optimizer!r} is not supported with NequIP; choose from {OPTIMIZERS}")
        relax_size, energy_size = NEQUIP_PLANS[name]
        self.name, self.device, self.optimizer = name, device, optimizer
        self.relax_calc, relax_file = get_calculator(relax_size, device)
        self.energy_calc, energy_file = (get_calculator(energy_size, device) if energy_size else (None, None))
        self.description = f"relax {relax_file}" + (f", energy {energy_file}" if energy_file else "") \
            + f", optimizer {optimizer}"

    def relax(self, structure, fmax=0.1, steps=500, **kwargs):
        """Remove atom overlaps if needed, relax with the collapse guard, and on a collapse retry
        once with NequIP-OAM-S + FIRE from the same start. Energy from the plan's energy model."""
        from pymatgen.io.ase import AseAtomsAdaptor

        structure, pre = preoptimize(structure)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        retry = 0
        try:
            with _collapse_guard(self.relax_calc, atoms):
                atoms.calc = self.relax_calc
                n_steps = _optimize(_cell_filter(atoms), self.optimizer, fmax, steps)
        except CellCollapse as exc:
            from ase.optimize import FIRE
            retry = 1
            s_calc, _ = get_calculator("S", self.device)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            with _collapse_guard(s_calc, atoms):  # a second collapse propagates (relax failed)
                atoms.calc = s_calc
                opt = FIRE(_cell_filter(atoms), logfile=None, maxstep=0.1)
                opt.run(fmax=fmax, steps=2 * steps)
                n_steps = opt.nsteps
            _log(f"relaxation collapsed ({exc}); retried with NequIP-OAM-S + FIRE")
        atoms.calc = None
        final = AseAtomsAdaptor.get_structure(atoms)
        atoms.calc = self.energy_calc or self.relax_calc
        energy = float(atoms.get_potential_energy())
        return {"final_structure": final, "trajectory": types.SimpleNamespace(energies=[energy]),
                "steps": n_steps, "preoptimized": pre["applied"], "retry": retry}


# ----------------------------------------------------------------------------- start-structure safety
# Lattice optimization occasionally returns overlapping atoms (closest pair down to 0 A). Started
# from such a structure, the ML force fields can collapse the cell (energy holes at unphysical
# distances), and the neighbor list of a collapsed cell runs for hours in C code that the
# SIGALRM time limit in core.py cannot interrupt. On the 41 such MP-20 structures, removing the
# overlaps first raised reference matches from 13-14 to 24-25 and removed all hangs.

OVERLAP_TRIGGER = 0.5   # pre-optimize when the closest pair is below this fraction of r_i + r_j
OVERLAP_TARGET = 0.7    # push pairs apart to at least this fraction
COLLAPSE_VOLUME = 2.0   # A^3 per atom; real crystals have >= ~5


class CellCollapse(RelaxBackendError):
    """The cell collapsed or degenerated during relaxation."""


def _covalent_radii(structure):
    from pymatgen.analysis.local_env import CovalentRadius
    import numpy as np
    return np.array([CovalentRadius.radius.get(el.symbol, 1.5) for el in structure.species])


def closest_pair_ratio(structure):
    """Smallest d_ij / (r_i + r_j) over all pairs (periodic images included), covalent radii."""
    import numpy as np
    rad = _covalent_radii(structure)
    d = structure.distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return float((d / (rad[:, None] + rad[None, :])).min())


def remove_overlaps(structure, target=OVERLAP_TARGET, max_iter=300, max_expand=1.5, seed=0):
    """Push apart atoms closer than target * (r_i + r_j) (pure geometry, no force field).

    Each overlapping pair moves apart along its separation vector by the missing distance,
    split between the two atoms; exactly coincident atoms get a random direction. If overlaps
    persist, the cell grows by 2% every 20 iterations (fractional coordinates kept), up to
    max_expand times its starting size. Takes milliseconds for SLICES-sized cells.
    """
    import numpy as np
    from pymatgen.core import Lattice, Structure
    rng = np.random.default_rng(seed)
    s = structure.copy()
    rad = _covalent_radii(s)
    scale = 1.0
    for it in range(max_iter):
        i, j, img, d = s.get_neighbor_list(target * 2 * rad.max())
        need = target * (rad[i] + rad[j])
        m = (d < need) & (i < j)
        if not m.any():
            break
        i, j, img, d, need = i[m], j[m], img[m], d[m], need[m]
        cart = s.cart_coords
        vec = cart[j] + img @ s.lattice.matrix - cart[i]
        tiny = d < 1e-3
        if tiny.any():
            rnd = rng.normal(size=(int(tiny.sum()), 3))
            vec[tiny] = rnd / np.linalg.norm(rnd, axis=1)[:, None] * 1e-3
            d = np.where(tiny, 1e-3, d)
        push = 0.525 * (need - d)[:, None] * vec / d[:, None]
        delta = np.zeros_like(cart)
        np.add.at(delta, i, -push)
        np.add.at(delta, j, push)
        cart = cart + delta
        lattice = s.lattice
        if it and it % 20 == 0 and scale * 1.02 <= max_expand:
            frac = lattice.get_fractional_coords(cart)
            lattice = Lattice(lattice.matrix * 1.02)
            cart = lattice.get_cartesian_coords(frac)
            scale *= 1.02
        s = Structure(lattice, s.species, cart, coords_are_cartesian=True)
    return s


def preoptimize(structure, trigger=OVERLAP_TRIGGER):
    """Niggli-reduce and remove overlaps when the closest pair is below trigger * (r_i + r_j);
    otherwise return the structure unchanged."""
    ratio = closest_pair_ratio(structure)
    if ratio >= trigger:
        return structure, {"applied": False, "ratio": ratio}
    s = structure
    try:
        s = s.get_reduced_structure("niggli")
    except Exception:  # degenerate cell: keep it, the overlap removal still applies
        pass
    s = remove_overlaps(s)
    return s, {"applied": True, "ratio_before": ratio, "ratio": closest_pair_ratio(s)}


class _collapse_guard:
    """Make an ASE calculator refuse collapsed or degenerate cells before evaluating them.

    Collapsed: below COLLAPSE_VOLUME per atom and below half the starting volume (a start that
    is already small, e.g. a one-atom He cell, may still relax). Degenerate: lattice lengths
    differing by more than 20x or an angle below 5 degrees (no MP-20 structure comes close).
    """

    def __init__(self, calc, atoms):
        self.calc, self.natoms = calc, len(atoms)
        self.min_volume = min(COLLAPSE_VOLUME, 0.5 * atoms.get_volume() / len(atoms))

    def __enter__(self):
        calc, natoms, min_volume, orig = self.calc, self.natoms, self.min_volume, self.calc.calculate
        self.orig = orig

        def calculate(atoms=None, *args, **kwargs):
            if atoms is not None:
                volume = atoms.get_volume() / natoms
                if not volume >= min_volume:
                    raise CellCollapse(f"cell collapsed to {volume:.2f} A^3/atom")
                lengths = atoms.cell.lengths()
                if lengths.max() > 20 * lengths.min() or min(atoms.cell.angles()) < 5:
                    raise CellCollapse("cell degenerated")
            return orig(atoms, *args, **kwargs)

        calc.calculate = calculate
        return self

    def __exit__(self, *exc):
        self.calc.calculate = self.orig
        return False


# ----------------------------------------------------------------------------- CLI

def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--relax-model", default="auto", help="relax_model to prepare (default: auto)")
    parser.add_argument("--prepare", action="store_true",
                        help="compile/cache the models and run a test relaxation")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]}  nequip installed: {nequip_available()}")
    if nequip_available():
        import torch
        cuda = torch.cuda.is_available()
        print(f"torch {torch.__version__}  CUDA available: {cuda}"
              + (f"  GPU: {torch.cuda.get_device_name(0)}" if cuda else "")
              + f"  OpenEquivariance: {importlib.util.find_spec('openequivariance') is not None}")
    name, device = resolve_relax_model(args.relax_model)
    print(f"relax_model={args.relax_model!r} resolves to {name!r} on {device}")
    if not args.prepare:
        return
    t0 = time.time()
    relaxer, name = make_relaxer(args.relax_model)
    print(f"ready in {time.time() - t0:.0f} s: {name}"
          + (f" ({relaxer.description})" if hasattr(relaxer, "description") else ""))
    if relaxer is None:
        return
    from pymatgen.core import Lattice, Structure
    struct = Structure(Lattice.cubic(4.3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    t0 = time.time()
    res = relaxer.relax(struct, fmax=0.2, steps=100)
    final = res["final_structure"]
    print(f"test relaxation (CsCl-type NaCl): {time.time() - t0:.2f} s, "
          f"E = {res['trajectory'].energies[-1] / len(final):.4f} eV/atom, a = {final.lattice.a:.3f} A")


if __name__ == "__main__":
    _main()
