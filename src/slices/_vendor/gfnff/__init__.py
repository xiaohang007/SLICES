# This file is part of gfnff.
# SPDX-Identifier: LGPL-3.0-or-later
"""
Python bindings for the GFN-FF force-field library.

Public API
----------
GFNFFCalculator
    Low-level interface (Bohr / Hartree units).
GFNFF
    ASE Calculator subclass (Angstrom / eV units).
    Available when ASE is installed; raises ImportError otherwise.
"""

from .calculator import GFNFFCalculator
from .topology import GFNFFTopology

try:
    from .ase_calculator import GFNFF
except ImportError:
    pass  # ASE not installed; GFNFF unavailable

__version__ = "0.0.1"
__all__ = ["GFNFFCalculator", "GFNFF", "GFNFFTopology"]
