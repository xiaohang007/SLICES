# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com

import os
import csv
import gc
import sqlite3
from collections import defaultdict

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

os.environ["OMP_NUM_THREADS"] = "1"

# Matching tolerances
ltol = 0.2
stol = 0.3
angle_tol = 5

DB_PATH = os.environ.get("STRUCTURE_DB_PATH", "../structure_database.db")


def _load_candidates(cursor, comp, cache):
    if comp in cache:
        return cache[comp]

    cursor.execute("SELECT primitive_cif FROM structures WHERE composition = ?", (comp,))
    rows = cursor.fetchall()
    candidates = []
    for (cif_string,) in rows:
        try:
            candidates.append(Structure.from_str(cif_string, "cif"))
        except Exception:
            continue

    cache[comp] = candidates
    return candidates


def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: structure database not found at {DB_PATH}")
        return

    if not os.path.exists("temp_splited.csv"):
        print("No temp_splited.csv found.")
        return

    with open("temp_splited.csv", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    if not lines:
        print("temp_splited.csv is empty, nothing to do.")
        return

    # Build StructureMatcher once
    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    candidate_cache = {}

    for line_idx, line in enumerate(lines, start=1):
        row = line.strip().split(",")
        if not row:
            continue

        try:
            poscar_input = row[-1].replace('\\n', '\n')
            query_struc = Structure.from_str(poscar_input, fmt="poscar")
            try:
                finder = SpacegroupAnalyzer(query_struc)
                query_struc = finder.get_primitive_standard_structure()
            except Exception:
                pass
        except Exception as e:
            with open("suspect_rows.csv", "a", encoding='utf-8', newline='') as fsus:
                writer_suspect = csv.writer(fsus)
                writer_suspect.writerow(row + ["POSCAR parse error", repr(e)])
            continue

        comp_query = query_struc.composition.reduced_formula
        candidates = _load_candidates(cursor, comp_query, candidate_cache)

        novelty = 1
        for candidate in candidates:
            try:
                if sm.fit(candidate, query_struc):
                    novelty = 0
                    break
            except Exception as e_fit:
                with open("suspect_rows.csv", "a", encoding='utf-8', newline='') as fsus:
                    writer_suspect = csv.writer(fsus)
                    writer_suspect.writerow(
                        row + [f"CandidateComp={candidate.composition}", f"sm.fit error: {repr(e_fit)}"]
                    )

        with open("result2.csv", "a", encoding='utf-8', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(row + [str(novelty)])

        del query_struc

    conn.close()
    gc.collect()


if __name__ == "__main__":
    main()
