# -*- coding: utf-8 -*-
"""Force fields for the final relaxation step of SLICES decoding.

``SLICES(relax_model=...)`` accepts:

* ``"auto"`` (default): NequIP-OAM-L on a usable CUDA GPU. Without a GPU, relax with
  NequIP-OAM-S and take the energy from a NequIP-OAM-L single point (about as fast as the
  former M3GNet relaxer on CPU).
* ``"nequip-l"``: relax with NequIP-OAM-L.
* ``"nequip-s+l"`` / ``"nequip-m+l"``: relax with NequIP-OAM-S / -M, energy from NequIP-OAM-L.
* ``None`` or ``"none"``: no relaxer.

All NequIP variants report the NequIP-OAM-L energy, so results from GPU and CPU machines are
on the same energy scale.

Environment variables:

* ``SLICES_RELAX_MODEL``: used when ``relax_model="auto"``, e.g. ``nequip-s+l`` or ``nequip-m+l``.
* ``SLICES_DEVICE``: ``cpu`` or ``cuda``, overrides GPU detection.
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
    """``cuda`` if a CUDA GPU is visible to torch, else ``cpu``; ``SLICES_DEVICE`` overrides."""
    forced = os.environ.get("SLICES_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    if forced:
        raise ValueError(f"SLICES_DEVICE must be 'cpu' or 'cuda', got {forced!r}")
    import torch
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def resolve_relax_model(relax_model):
    """Return ``(name, device, from_auto)`` for a ``relax_model`` value."""
    if relax_model is None:
        return "none", None, False
    name = str(relax_model).strip().lower()
    from_auto = name == "auto"
    if from_auto:
        name = os.environ.get("SLICES_RELAX_MODEL", "auto").strip().lower() or "auto"
    if name == "m3gnet":
        raise ValueError("M3GNet support has been removed; use relax_model='auto' or one of "
                         f"{tuple(NEQUIP_PLANS)}")
    if name not in RELAX_MODELS:
        raise ValueError(f"Unknown relax_model {relax_model!r}; choose from {RELAX_MODELS}")
    if name == "auto":
        if not nequip_available():
            raise ImportError(f"relax_model='auto' needs torch and nequip ({_nequip_install_hint()})")
        device = choose_device()
        return ("nequip-l" if device == "cuda" else "nequip-s+l"), device, True
    if name in NEQUIP_PLANS:
        if not nequip_available():
            raise ImportError(f"relax_model={name!r} needs torch and nequip ({_nequip_install_hint()})")
        return name, choose_device(), from_auto
    return name, None, from_auto


def make_relaxer(relax_model="auto", optimizer="BFGS"):
    """Build the relaxer for ``relax_model``. Returns ``(relaxer or None, resolved name)``."""
    name, device, from_auto = resolve_relax_model(relax_model)
    if name == "none":
        return None, "none"
    try:
        relaxer = NequIPRelaxer(name, device, optimizer)
    except Exception as exc:
        if not (from_auto and device == "cuda"):
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
    return NequIPCalculator.from_compiled_model(path, device=device)


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
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.relax_calc
        n_steps = _optimize(_cell_filter(atoms), self.optimizer, fmax, steps)
        atoms.calc = None
        final = AseAtomsAdaptor.get_structure(atoms)
        atoms.calc = self.energy_calc or self.relax_calc
        energy = float(atoms.get_potential_energy())
        return {"final_structure": final, "trajectory": types.SimpleNamespace(energies=[energy]),
                "steps": n_steps}


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
    name, device, _ = resolve_relax_model(args.relax_model)
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
