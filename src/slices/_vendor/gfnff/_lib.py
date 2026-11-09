# This file is part of gfnff.
# SPDX-Identifier: LGPL-3.0-or-later
"""
ctypes loader and raw C API prototypes for libgfnff.

Public functions from gfnff_interface_c.h are wrapped here.
Units match the C API exactly: Bohr, Hartree, Eh/Bohr.
"""

import ctypes
import ctypes.util
import pathlib

import numpy as np
import numpy.ctypeslib as npct


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library() -> ctypes.CDLL:
    # 1. Wheel / editable install: .so lives next to this file
    pkg_dir = pathlib.Path(__file__).parent
    candidates = (
        list(pkg_dir.glob("libgfnff.so*"))   # Linux: libgfnff.so.0.0.1
        + list(pkg_dir.glob("libgfnff*.dylib"))  # macOS: libgfnff.0.0.1.dylib
        + list(pkg_dir.glob("gfnff.dll"))
    )
    if candidates:
        return ctypes.CDLL(str(candidates[0]))

    # 2. LD_LIBRARY_PATH / system library
    name = ctypes.util.find_library("gfnff")
    if name:
        return ctypes.CDLL(name)

    raise OSError(
        "libgfnff shared library not found. "
        "Install the gfnff package or set LD_LIBRARY_PATH to the directory "
        "containing libgfnff.so."
    )


_lib = _load_library()


# ---------------------------------------------------------------------------
# Opaque calculator struct
# ---------------------------------------------------------------------------

class _CGFNFFCalculator(ctypes.Structure):
    """Mirror of ``typedef struct { void *ptr; } c_gfnff_calculator;``"""
    _fields_ = [("ptr", ctypes.c_void_p)]


# ---------------------------------------------------------------------------
# Convenience ndpointer types
# ---------------------------------------------------------------------------

# double xyz[nat][3]  or  double gradient[nat][3]  -- shape checked at runtime
_xyz_type = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")

# int at[nat]
_at_type = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

# int matrix[n][m]
_int_matrix_type = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

# double matrix[n][m]
_double_matrix_type = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")

# double lattice[3][3]
_lattice_type = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


# ---------------------------------------------------------------------------
# Function prototypes
# ---------------------------------------------------------------------------

# c_gfnff_calculator c_gfnff_calculator_init(
#     int nat, int *at, double (*xyz)[3],
#     int ichrg, int printlevel, const char *solvent);
_lib.c_gfnff_calculator_init.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init.argtypes = [
    ctypes.c_int,    # nat
    _at_type,        # at[nat]
    _xyz_type,       # xyz[nat][3]
    ctypes.c_int,    # ichrg
    ctypes.c_int,    # printlevel
    ctypes.c_char_p, # solvent (null-terminated)
]

# c_gfnff_calculator c_gfnff_calculator_init_nbf(
#     int nat, int *at, int (*nbf)[20],
#     int ichrg, int printlevel);
_lib.c_gfnff_calculator_init_nbf.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_nbf.argtypes = [
    ctypes.c_int,    # nat
    _at_type,        # at[nat]
    _int_matrix_type, # nbf[nat][20], C-contiguous
    ctypes.c_int,    # ichrg
    ctypes.c_int,    # printlevel
]

# c_gfnff_calculator c_gfnff_calculator_init_nbf_param(
#     int nat, int *at, int (*nbf)[20],
#     int ichrg, int printlevel, const char *parametrisation);
_lib.c_gfnff_calculator_init_nbf_param.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_nbf_param.argtypes = [
    ctypes.c_int,     # nat
    _at_type,         # at[nat]
    _int_matrix_type, # nbf[nat][20], C-contiguous
    ctypes.c_int,     # ichrg
    ctypes.c_int,     # printlevel
    ctypes.c_char_p,  # parameter file path
]

# c_gfnff_calculator c_gfnff_calculator_init_periodic_edges(
#     int nat, int *at, double (*xyz)[3],
#     int nedge, int *edge_i, int *edge_j, int (*edge_shift)[3],
#     double lattice[3][3], int npbc, int ichrg, int printlevel);
_lib.c_gfnff_calculator_init_periodic_edges.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_periodic_edges.argtypes = [
    ctypes.c_int,     # nat
    _at_type,         # at[nat]
    _xyz_type,        # xyz[nat][3]
    ctypes.c_int,     # nedge
    _at_type,         # edge_i[nedge]
    _at_type,         # edge_j[nedge]
    _int_matrix_type, # edge_shift[nedge][3], C-contiguous
    _lattice_type,    # lattice[3][3]
    ctypes.c_int,     # npbc
    ctypes.c_int,     # ichrg
    ctypes.c_int,     # printlevel
]

# c_gfnff_calculator c_gfnff_calculator_init_periodic_edges_param(
#     int nat, int *at, double (*xyz)[3],
#     int nedge, int *edge_i, int *edge_j, int (*edge_shift)[3],
#     double lattice[3][3], int npbc, int ichrg, int printlevel,
#     const char *parametrisation);
_lib.c_gfnff_calculator_init_periodic_edges_param.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_periodic_edges_param.argtypes = [
    ctypes.c_int,     # nat
    _at_type,         # at[nat]
    _xyz_type,        # xyz[nat][3]
    ctypes.c_int,     # nedge
    _at_type,         # edge_i[nedge]
    _at_type,         # edge_j[nedge]
    _int_matrix_type, # edge_shift[nedge][3], C-contiguous
    _lattice_type,    # lattice[3][3]
    ctypes.c_int,     # npbc
    ctypes.c_int,     # ichrg
    ctypes.c_int,     # printlevel
    ctypes.c_char_p,  # parameter file path
]

# c_gfnff_calculator c_gfnff_calculator_init_pbc(
#     int nat, int *at, double (*xyz)[3],
#     int ichrg, int printlevel,
#     double lattice[3][3], int npbc);
_lib.c_gfnff_calculator_init_pbc.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_pbc.argtypes = [
    ctypes.c_int,    # nat
    _at_type,        # at[nat]
    _xyz_type,       # xyz[nat][3]
    ctypes.c_int,    # ichrg
    ctypes.c_int,    # printlevel
    _lattice_type,   # lattice[3][3]
    ctypes.c_int,    # npbc
]

# c_gfnff_calculator c_gfnff_calculator_init_ex(
#     int nat, int *at, double (*xyz)[3],
#     int ichrg, int printlevel, const char *solvent,
#     double (*lattice)[3],  // or NULL -> non-periodic
#     int npbc,
#     int *fraglist,         // or NULL -> automatic fragmentation
#     double *refq);         // or NULL -> no reference charges
_lib.c_gfnff_calculator_init_ex.restype = _CGFNFFCalculator
_lib.c_gfnff_calculator_init_ex.argtypes = [
    ctypes.c_int,    # nat
    _at_type,        # at[nat]
    _xyz_type,       # xyz[nat][3]
    ctypes.c_int,    # ichrg
    ctypes.c_int,    # printlevel
    ctypes.c_char_p, # solvent (null-terminated)
    ctypes.c_void_p, # lattice[3][3] or NULL
    ctypes.c_int,    # npbc
    ctypes.c_void_p, # fraglist[nat] or NULL
    ctypes.c_void_p, # refq[nat] or NULL
]

# void c_gfnff_calculator_singlepoint(
#     c_gfnff_calculator *calculator,
#     int nat, int *at, double (*xyz)[3],
#     double *energy, double (*gradient)[3],
#     double sigma[3][3],
#     double (*lattice)[3],   // NULL to reuse stored lattice
#     int *iostat);
_lib.c_gfnff_calculator_singlepoint.restype = None
_lib.c_gfnff_calculator_singlepoint.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),  # calculator
    ctypes.c_int,                        # nat
    _at_type,                            # at[nat]
    _xyz_type,                           # xyz[nat][3]
    ctypes.POINTER(ctypes.c_double),     # energy (out)
    _xyz_type,                           # gradient[nat][3] (out)
    _lattice_type,                       # sigma[3][3] (out); zero for non-PBC
    ctypes.c_void_p,                     # lattice[3][3] or NULL
    ctypes.POINTER(ctypes.c_int),        # iostat (out)
]

# void c_gfnff_calculator_results(c_gfnff_calculator *calculator, int iunit);
_lib.c_gfnff_calculator_results.restype = None
_lib.c_gfnff_calculator_results.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),  # calculator
    ctypes.c_int,                        # iunit (6 = Fortran stdout)
]

# void c_gfnff_topology_sizes(
#     c_gfnff_calculator *calculator,
#     int *nbond_blist, int *nbond_vbond, int *nangl, int *iostat);
_lib.c_gfnff_topology_sizes.restype = None
_lib.c_gfnff_topology_sizes.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]

# void c_gfnff_topology_copy(
#     c_gfnff_calculator *calculator,
#     int blist[nbond_blist][2],
#     double vbond[nbond_vbond][6],
#     int alist[nangl][3],
#     double vangl[nangl][2],
#     int *iostat);
_lib.c_gfnff_topology_copy.restype = None
_lib.c_gfnff_topology_copy.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),
    _int_matrix_type,
    _double_matrix_type,
    _int_matrix_type,
    _double_matrix_type,
    ctypes.POINTER(ctypes.c_int),
]

# void c_gfnff_topology_copy_pbc(
#     c_gfnff_calculator *calculator,
#     int blist[nbond_blist][2],
#     int bshift[nbond_blist][3],
#     double vbond[nbond_vbond][6],
#     int alist[nangl][3],
#     int ashift[nangl][6],
#     double vangl[nangl][2],
#     int *iostat);
_lib.c_gfnff_topology_copy_pbc.restype = None
_lib.c_gfnff_topology_copy_pbc.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),
    _int_matrix_type,
    _int_matrix_type,
    _double_matrix_type,
    _int_matrix_type,
    _int_matrix_type,
    _double_matrix_type,
    ctypes.POINTER(ctypes.c_int),
]

# void c_gfnff_calculator_deallocate(c_gfnff_calculator *calculator);
_lib.c_gfnff_calculator_deallocate.restype = None
_lib.c_gfnff_calculator_deallocate.argtypes = [
    ctypes.POINTER(_CGFNFFCalculator),  # calculator
]
