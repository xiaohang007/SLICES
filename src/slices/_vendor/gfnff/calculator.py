# This file is part of gfnff.
# SPDX-Identifier: LGPL-3.0-or-later
"""
Low-level Python wrapper around the GFN-FF C API.

All quantities use the native C API units:
  - positions / lattice : Bohr
  - energy              : Hartree
  - gradient            : Eh / Bohr
"""

import ctypes

import numpy as np

from ._lib import _CGFNFFCalculator, _lib


class GFNFFCalculator:
    """Wraps the GFN-FF C API for a single molecular or periodic system.

    Parameters
    ----------
    numbers:
        Atomic numbers, shape ``(nat,)``.
    positions_bohr:
        Cartesian coordinates in Bohr, shape ``(nat, 3)``, C-contiguous.
    charge:
        Total molecular charge (integer).
    printlevel:
        Verbosity of Fortran output (0 = silent, 1 = errors only,
        2 = informational, 3 = verbose).
    solvent:
        Implicit solvent name (e.g. ``"h2o"``, ``"acetone"``).
        Empty string or ``None`` disables solvation.
    lattice:
        Lattice vectors in Bohr, shape ``(3, 3)``, C-contiguous.
        Each *row* is one lattice vector (same convention as the C API).
        ``None`` for non-periodic systems.
    npbc:
        Number of periodic dimensions (0–3).  Ignored when
        ``lattice`` is ``None``.
    """

    def __init__(
        self,
        numbers,
        positions_bohr,
        *,
        charge: int = 0,
        printlevel: int = 0,
        solvent: str = "",
        lattice=None,
        npbc: int = 0,
        fragments=None,
        ref_charges=None,
    ):
        """
        fragments:
            Optional per-atom fragment index, shape ``(nat,)`` (int). When given,
            GFN-FF forms no bonds between atoms of differing fragments, enforcing
            a host-defined fragmentation instead of the automatic detection.
        ref_charges:
            Optional per-atom reference charges, shape ``(nat,)`` (float). Summed
            over each fragment to set the per-fragment EEQ charge constraint.
            No charge model is invoked; the caller supplies the array directly.
        """
        at = np.asarray(numbers, dtype=np.int32)
        xyz = np.ascontiguousarray(positions_bohr, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"positions_bohr must have shape (nat, 3), got {xyz.shape}"
            )
        if at.shape[0] != xyz.shape[0]:
            raise ValueError("numbers and positions_bohr must have the same length")

        nat = ctypes.c_int(len(at))
        solvent_bytes = (solvent or "").encode()

        # validate and keep references to optional bundle arrays alive
        frag_arr = None
        frag_ptr = None
        if fragments is not None:
            frag_arr = np.ascontiguousarray(fragments, dtype=np.int32)
            if frag_arr.shape != (len(at),):
                raise ValueError(
                    f"fragments must have shape ({len(at)},), got {frag_arr.shape}"
                )
            frag_ptr = frag_arr.ctypes.data_as(ctypes.c_void_p)

        refq_arr = None
        refq_ptr = None
        if ref_charges is not None:
            refq_arr = np.ascontiguousarray(ref_charges, dtype=np.float64)
            if refq_arr.shape != (len(at),):
                raise ValueError(
                    f"ref_charges must have shape ({len(at)},), got {refq_arr.shape}"
                )
            refq_ptr = refq_arr.ctypes.data_as(ctypes.c_void_p)

        if fragments is not None or ref_charges is not None:
            # extended path: carries the optional host-supplied bundle hints
            if lattice is not None:
                lat = np.ascontiguousarray(lattice, dtype=np.float64)
                if lat.shape != (3, 3):
                    raise ValueError(
                        f"lattice must have shape (3, 3), got {lat.shape}"
                    )
                lattice_ptr = lat.ctypes.data_as(ctypes.c_void_p)
            else:
                lattice_ptr = None
            self._handle = _lib.c_gfnff_calculator_init_ex(
                nat, at, xyz,
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
                solvent_bytes,
                lattice_ptr,
                ctypes.c_int(npbc),
                frag_ptr,
                refq_ptr,
            )
        elif lattice is not None:
            lat = np.ascontiguousarray(lattice, dtype=np.float64)
            if lat.shape != (3, 3):
                raise ValueError(f"lattice must have shape (3, 3), got {lat.shape}")
            self._handle = _lib.c_gfnff_calculator_init_pbc(
                nat, at, xyz,
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
                lat,
                ctypes.c_int(npbc),
            )
        else:
            self._handle = _lib.c_gfnff_calculator_init(
                nat, at, xyz,
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
                solvent_bytes,
            )

        if self._handle.ptr is None:
            raise RuntimeError(
                "GFN-FF initialization failed (c_gfnff_calculator_init returned NULL). "
                "Check printlevel > 0 for error details."
            )
        self._alive = True
        self._topology_source = "xyz_initialized_stock_gfnff"
        self._parametrisation = ""

    # ------------------------------------------------------------------

    @classmethod
    def from_nbf(
        cls,
        numbers,
        neighbor_table_1based,
        *,
        charge: int = 0,
        printlevel: int = 0,
        parametrisation: str = "",
    ):
        """Initialize from a topology-only legacy NBF neighbor table.

        ``neighbor_table_1based`` must have shape ``(20, nat)``. Rows 0-18
        contain 1-based neighbor ids and row 19 contains the per-atom degree.
        No physical coordinates are accepted by this initializer.
        """
        at = np.asarray(numbers, dtype=np.int32)
        nbf = np.asarray(neighbor_table_1based, dtype=np.int32)
        if nbf.shape != (20, len(at)):
            raise ValueError(
                f"neighbor_table_1based must have shape (20, {len(at)}), got {nbf.shape}"
            )
        degrees = nbf[19]
        if np.any(degrees < 0) or np.any(degrees > 19):
            raise ValueError("NBF degree row must contain values in [0, 19]")
        for atom_index, degree in enumerate(degrees):
            listed = nbf[: int(degree), atom_index]
            if np.any(listed < 1) or np.any(listed > len(at)):
                raise ValueError(
                    f"NBF neighbor ids for atom {atom_index + 1} must be 1-based and <= nat"
                )

        obj = cls.__new__(cls)
        nat = ctypes.c_int(len(at))
        # Fortran sees c_nbf(20,nat); C/Python provide row-major nbf[nat][20].
        nbf_c = np.ascontiguousarray(nbf.T, dtype=np.int32)
        param_path = str(parametrisation or "")
        if param_path:
            obj._handle = _lib.c_gfnff_calculator_init_nbf_param(
                nat,
                at,
                nbf_c,
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
                param_path.encode(),
            )
        else:
            obj._handle = _lib.c_gfnff_calculator_init_nbf(
                nat,
                at,
                nbf_c,
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
            )
        if obj._handle.ptr is None:
            raise RuntimeError(
                "GFN-FF topology-only NBF initialization failed. "
                "Check printlevel > 0 for error details."
            )
        obj._alive = True
        obj._topology_source = "topology_only_nbf_v1"
        obj._parametrisation = param_path
        return obj

    # ------------------------------------------------------------------

    @classmethod
    def from_periodic_edges(
        cls,
        numbers,
        edge_i_1based,
        edge_j_1based,
        edge_shift,
        *,
        positions_bohr=None,
        lattice_bohr=None,
        charge: int = 0,
        printlevel: int = 0,
        npbc: int = 3,
        parametrisation: str = "",
    ):
        """Initialize from a topology-only periodic edge list.

        The host-provided edges are the only topology source. ``edge_shift``
        gives the integer lattice image of ``edge_j`` relative to ``edge_i``.
        If no geometry is supplied, deterministic well-spaced coordinates and
        a large orthogonal cell are used only to keep geometry-dependent setup
        numerically defined.
        """
        at = np.asarray(numbers, dtype=np.int32)
        edge_i = np.asarray(edge_i_1based, dtype=np.int32)
        edge_j = np.asarray(edge_j_1based, dtype=np.int32)
        shift = np.ascontiguousarray(edge_shift, dtype=np.int32)
        if edge_i.shape != edge_j.shape:
            raise ValueError("edge_i_1based and edge_j_1based must have the same shape")
        if edge_i.ndim != 1:
            raise ValueError("edge_i_1based and edge_j_1based must be 1D arrays")
        if shift.shape != (len(edge_i), 3):
            raise ValueError(
                f"edge_shift must have shape ({len(edge_i)}, 3), got {shift.shape}"
            )
        if np.any(edge_i < 1) or np.any(edge_i > len(at)):
            raise ValueError("edge_i_1based entries must be in [1, nat]")
        if np.any(edge_j < 1) or np.any(edge_j > len(at)):
            raise ValueError("edge_j_1based entries must be in [1, nat]")

        if positions_bohr is None:
            xyz = np.zeros((len(at), 3), dtype=np.float64)
            side = max(1, int(np.ceil(len(at) ** (1.0 / 3.0))))
            spacing = 4.0
            for index in range(len(at)):
                xyz[index, 0] = spacing * (index % side)
                xyz[index, 1] = spacing * ((index // side) % side)
                xyz[index, 2] = spacing * (index // (side * side))
            xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        else:
            xyz = np.ascontiguousarray(positions_bohr, dtype=np.float64)
            if xyz.shape != (len(at), 3):
                raise ValueError(
                    f"positions_bohr must have shape ({len(at)}, 3), got {xyz.shape}"
                )

        if lattice_bohr is None:
            lat = np.ascontiguousarray(np.eye(3, dtype=np.float64) * 100.0)
        else:
            lat = np.ascontiguousarray(lattice_bohr, dtype=np.float64)
            if lat.shape != (3, 3):
                raise ValueError(f"lattice_bohr must have shape (3, 3), got {lat.shape}")

        obj = cls.__new__(cls)
        nat = ctypes.c_int(len(at))
        param_path = str(parametrisation or "")
        if param_path:
            obj._handle = _lib.c_gfnff_calculator_init_periodic_edges_param(
                nat,
                at,
                xyz,
                ctypes.c_int(len(edge_i)),
                np.ascontiguousarray(edge_i, dtype=np.int32),
                np.ascontiguousarray(edge_j, dtype=np.int32),
                shift,
                lat,
                ctypes.c_int(npbc),
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
                param_path.encode(),
            )
        else:
            obj._handle = _lib.c_gfnff_calculator_init_periodic_edges(
                nat,
                at,
                xyz,
                ctypes.c_int(len(edge_i)),
                np.ascontiguousarray(edge_i, dtype=np.int32),
                np.ascontiguousarray(edge_j, dtype=np.int32),
                shift,
                lat,
                ctypes.c_int(npbc),
                ctypes.c_int(charge),
                ctypes.c_int(printlevel),
            )
        if obj._handle.ptr is None:
            raise RuntimeError(
                "GFN-FF topology-only periodic-edge initialization failed. "
                "Check printlevel > 0 for error details."
            )
        obj._alive = True
        obj._topology_source = "topology_only_periodic_edges_v1"
        obj._parametrisation = param_path
        return obj

    # ------------------------------------------------------------------

    def singlepoint(self, numbers, positions_bohr, lattice=None):
        """Compute energy, gradient and stress tensor for the given geometry.

        Parameters
        ----------
        numbers:
            Atomic numbers, shape ``(nat,)``.
        positions_bohr:
            Coordinates in Bohr, shape ``(nat, 3)``.
        lattice:
            Lattice vectors in Bohr, shape ``(3, 3)``.  Must be provided
            for periodic systems (same ``npbc`` as at initialization).

        Returns
        -------
        energy : float
            Total energy in Hartree.
        gradient : ndarray, shape ``(nat, 3)``
            Energy gradient in Eh / Bohr.
        sigma : ndarray, shape ``(3, 3)``
            Stress tensor in Hartree.  Zero for non-periodic systems.
        """
        self._check_alive()

        at = np.asarray(numbers, dtype=np.int32)
        xyz = np.ascontiguousarray(positions_bohr, dtype=np.float64)
        nat = len(at)

        energy_out = ctypes.c_double(0.0)
        gradient = np.zeros((nat, 3), dtype=np.float64, order="C")
        sigma = np.zeros((3, 3), dtype=np.float64, order="C")
        iostat = ctypes.c_int(0)

        if lattice is not None:
            lat = np.ascontiguousarray(lattice, dtype=np.float64)
            lattice_ptr = lat.ctypes.data_as(ctypes.c_void_p)
        else:
            lattice_ptr = None

        _lib.c_gfnff_calculator_singlepoint(
            ctypes.byref(self._handle),
            ctypes.c_int(nat),
            at,
            xyz,
            ctypes.byref(energy_out),
            gradient,
            sigma,
            lattice_ptr,
            ctypes.byref(iostat),
        )

        if iostat.value != 0:
            raise RuntimeError(
                f"GFN-FF singlepoint failed with iostat = {iostat.value}"
            )

        return float(energy_out.value), gradient, sigma

    # ------------------------------------------------------------------

    def print_results(self, iunit: int = 6):
        """Print the GFN-FF energy decomposition to a Fortran unit.

        Parameters
        ----------
        iunit:
            Fortran I/O unit number.  Use ``6`` for standard output.
        """
        self._check_alive()
        _lib.c_gfnff_calculator_results(
            ctypes.byref(self._handle),
            ctypes.c_int(iunit),
        )

    # ------------------------------------------------------------------

    def topology_sizes(self):
        """Return sizes for internal GFN-FF topology arrays.

        This exposes the existing xyz-initialized topology.  It does not create
        or accept a topology-only neighbor-list input.
        """
        self._check_alive()
        nbond_blist = ctypes.c_int(0)
        nbond_vbond = ctypes.c_int(0)
        nangl = ctypes.c_int(0)
        iostat = ctypes.c_int(0)
        _lib.c_gfnff_topology_sizes(
            ctypes.byref(self._handle),
            ctypes.byref(nbond_blist),
            ctypes.byref(nbond_vbond),
            ctypes.byref(nangl),
            ctypes.byref(iostat),
        )
        if iostat.value != 0:
            raise RuntimeError(
                f"GFN-FF topology size query failed with iostat = {iostat.value}"
            )
        return {
            "nbond_blist": int(nbond_blist.value),
            "nbond_vbond": int(nbond_vbond.value),
            "nangl": int(nangl.value),
        }

    # ------------------------------------------------------------------

    def topology_lists(self):
        """Return internal topology arrays in a legacy-list-compatible dict."""
        sizes = self.topology_sizes()
        nbond_blist = sizes["nbond_blist"]
        nbond_vbond = sizes["nbond_vbond"]
        nangl = sizes["nangl"]
        topology_source = getattr(
            self, "_topology_source", "xyz_initialized_stock_gfnff"
        )
        backend = (
            "pprcht_gfnff_topology_only_nbf_export"
            if topology_source == "topology_only_nbf_v1"
            else "pprcht_gfnff_xyz_topology_export"
        )

        blist = np.zeros((max(nbond_blist, 1), 2), dtype=np.int32, order="C")
        vbond = np.zeros((max(nbond_vbond, 1), 6), dtype=np.float64, order="C")
        alist = np.zeros((max(nangl, 1), 3), dtype=np.int32, order="C")
        vangl = np.zeros((max(nangl, 1), 2), dtype=np.float64, order="C")
        iostat = ctypes.c_int(0)

        _lib.c_gfnff_topology_copy(
            ctypes.byref(self._handle),
            blist,
            vbond,
            alist,
            vangl,
            ctypes.byref(iostat),
        )
        if iostat.value != 0:
            raise RuntimeError(
                f"GFN-FF topology copy failed with iostat = {iostat.value}"
            )
        return {
            "blist": blist[:nbond_blist].tolist(),
            "vbond": vbond[:nbond_vbond].tolist(),
            "alist": alist[:nangl].tolist(),
            "vangl": vangl[:nangl].tolist(),
            "metadata": {
                "backend": backend,
                "topology_source": topology_source,
                "bond_source": "neigh",
                "angle_source": "topo",
                "sizes": sizes,
                "parametrisation": getattr(self, "_parametrisation", ""),
                "parametrisation_source": (
                    "external_file"
                    if getattr(self, "_parametrisation", "")
                    else "internal_modern"
                ),
            },
        }

    # ------------------------------------------------------------------

    def topology_lists_pbc(self):
        """Return topology arrays plus integer PBC shifts for bonds/angles."""
        sizes = self.topology_sizes()
        nbond_blist = sizes["nbond_blist"]
        nbond_vbond = sizes["nbond_vbond"]
        nangl = sizes["nangl"]
        topology_source = getattr(
            self, "_topology_source", "xyz_initialized_stock_gfnff"
        )
        if topology_source == "topology_only_nbf_v1":
            backend = "pprcht_gfnff_topology_only_nbf_export"
        elif topology_source == "topology_only_periodic_edges_v1":
            backend = "pprcht_gfnff_topology_only_periodic_edges_export"
        else:
            backend = "pprcht_gfnff_xyz_topology_export"

        blist = np.zeros((max(nbond_blist, 1), 2), dtype=np.int32, order="C")
        bshift = np.zeros((max(nbond_blist, 1), 3), dtype=np.int32, order="C")
        vbond = np.zeros((max(nbond_vbond, 1), 6), dtype=np.float64, order="C")
        alist = np.zeros((max(nangl, 1), 3), dtype=np.int32, order="C")
        ashift = np.zeros((max(nangl, 1), 6), dtype=np.int32, order="C")
        vangl = np.zeros((max(nangl, 1), 2), dtype=np.float64, order="C")
        iostat = ctypes.c_int(0)

        _lib.c_gfnff_topology_copy_pbc(
            ctypes.byref(self._handle),
            blist,
            bshift,
            vbond,
            alist,
            ashift,
            vangl,
            ctypes.byref(iostat),
        )
        if iostat.value != 0:
            raise RuntimeError(
                f"GFN-FF PBC topology copy failed with iostat = {iostat.value}"
            )
        return {
            "blist": blist[:nbond_blist].tolist(),
            "blist_shift": bshift[:nbond_blist].tolist(),
            "vbond": vbond[:nbond_vbond].tolist(),
            "alist": alist[:nangl].tolist(),
            "alist_shift": ashift[:nangl].tolist(),
            "vangl": vangl[:nangl].tolist(),
            "metadata": {
                "backend": backend,
                "topology_source": topology_source,
                "bond_source": "neigh",
                "angle_source": "topo",
                "sizes": sizes,
                "pbc_shift_export": True,
                "parametrisation": getattr(self, "_parametrisation", ""),
                "parametrisation_source": (
                    "external_file"
                    if getattr(self, "_parametrisation", "")
                    else "internal_modern"
                ),
            },
        }

    # ------------------------------------------------------------------

    def _check_alive(self):
        if not self._alive or self._handle.ptr is None:
            raise RuntimeError("GFNFFCalculator has already been deallocated.")

    def deallocate(self):
        """Explicitly free Fortran-side memory."""
        if getattr(self, "_alive", False) and self._handle.ptr is not None:
            _lib.c_gfnff_calculator_deallocate(ctypes.byref(self._handle))
            self._alive = False

    def __del__(self):
        self.deallocate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.deallocate()
