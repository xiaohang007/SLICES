# This file is part of gfnff.
# SPDX-Identifier: LGPL-3.0-or-later
"""Topology export helpers for the Python GFN-FF bindings.

This module exports topology produced by the normal xyz-based GFN-FF
initializer and by the experimental topology-only NBF initializer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calculator import GFNFFCalculator


@dataclass
class GFNFFTopology:
    """Small wrapper around legacy-list-compatible topology arrays."""

    blist: list
    vbond: list
    alist: list
    vangl: list
    metadata: dict
    blist_shift: list | None = None
    alist_shift: list | None = None

    @classmethod
    def from_calculator(cls, calculator: GFNFFCalculator) -> "GFNFFTopology":
        payload = calculator.topology_lists()
        return cls(
            blist=payload["blist"],
            vbond=payload["vbond"],
            alist=payload["alist"],
            vangl=payload["vangl"],
            metadata=payload["metadata"],
            blist_shift=payload.get("blist_shift"),
            alist_shift=payload.get("alist_shift"),
        )

    @classmethod
    def from_nbf(
        cls,
        numbers,
        neighbor_table_1based,
        *,
        charge: int = 0,
        printlevel: int = 0,
        parametrisation: str = "",
    ) -> "GFNFFTopology":
        calculator = GFNFFCalculator.from_nbf(
            numbers,
            neighbor_table_1based,
            charge=charge,
            printlevel=printlevel,
            parametrisation=parametrisation,
        )
        try:
            return cls.from_calculator(calculator)
        finally:
            calculator.deallocate()

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
    ) -> "GFNFFTopology":
        calculator = GFNFFCalculator.from_periodic_edges(
            numbers,
            edge_i_1based,
            edge_j_1based,
            edge_shift,
            positions_bohr=positions_bohr,
            lattice_bohr=lattice_bohr,
            charge=charge,
            printlevel=printlevel,
            npbc=npbc,
            parametrisation=parametrisation,
        )
        try:
            payload = calculator.topology_lists_pbc()
            return cls(
                blist=payload["blist"],
                vbond=payload["vbond"],
                alist=payload["alist"],
                vangl=payload["vangl"],
                metadata=payload["metadata"],
                blist_shift=payload.get("blist_shift"),
                alist_shift=payload.get("alist_shift"),
            )
        finally:
            calculator.deallocate()

    def to_legacy_lists(self) -> dict:
        payload = {
            "blist": self.blist,
            "vbond": self.vbond,
            "alist": self.alist,
            "vangl": self.vangl,
            "metadata": self.metadata,
        }
        if self.blist_shift is not None:
            payload["blist_shift"] = self.blist_shift
        if self.alist_shift is not None:
            payload["alist_shift"] = self.alist_shift
        return payload
