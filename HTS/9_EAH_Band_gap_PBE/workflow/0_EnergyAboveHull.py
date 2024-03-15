#!/usr/bin/env python

__author__ = "Shyue Ping Ong"  "Hang Xiao"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__status__ = "Production"
__version__ = "1.0"
__date__ = "Jan 13, 2019"
# -*- coding: utf-8 -*-
# modified Hang Xiao 2023.04
# xiaohang07@live.cn
import argparse
import sys,re
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry
from monty.serialization import loadfn, dumpfn
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#This initializes the REST adaptor.

def get_num_atoms(formula):
    s = re.findall('([A-Z][a-z]?)([0-9]?\.?[0-9]*)', formula)
    #print(s)
    num_atoms = 0
    for sa, sb in s:
        if sb=='':
            sb=1
        num_atoms += int(sb)
    return num_atoms

def powerset(s):
    ret = []
    x = len(s)
    for i in range(1 << x):
        ret.append([s[j] for j in range(x) if (i & (1 << j))])
    return ret


def formulaExpander(formula):
    while len(re.findall('\(\w*\)', formula)) > 0:
        parenthetical = re.findall('\(\w*\)[0-9]+', formula)
        for i in parenthetical:
            p = re.findall('[0-9]+', str(re.findall('\)[0-9]+', i)))
            j = re.findall('[A-Z][a-z]*[0-9]*', i)
            for n in range(0, len(j)):
                numero = re.findall('[0-9]+', j[n])
                if len(numero) != 0:
                    for k in numero:
                        nu = re.sub(k, str(int(int(k) * int(p[0]))), j[n])
                else:
                    nu = re.sub(j[n], j[n] + p[0], j[n])
                j[n] = nu
        newphrase = ""
        for m in j:
            newphrase += str(m)
        formula = formula.replace(i, newphrase)
        if (len((re.findall('\(\w*\)[0-9]+', formula))) == 0) and (len(re.findall('\(\w*\)', formula)) != 0):
            formula = formula.replace('(', '')
            formula = formula.replace(')', '')
    return formula

drone = VaspToComputedEntryDrone()
entry = drone.assimilate("EFormation")
s = Structure.from_file("./EFormation/CONTCAR")
finder = SpacegroupAnalyzer(s)

space_group_number=finder.get_space_group_number()
compat = MaterialsProject2020Compatibility()
entry = compat.process_entry(entry)
if not entry:
	print("Calculation parameters are not consistent with Materials Project parameters.")
	sys.exit()


#This gets all entries belonging to the relevant system.
competitive_compositions=loadfn("../../7_EAH_prescreen/competitive_compositions.json.gz")
competitive_entries = []
for comp in competitive_compositions:
    flag=0
    competitive_comp = comp['composition']  # cannot use comp.composition like native MP object.
    for j in competitive_comp.elements:
        if j not in entry.composition.elements:
            flag=1
            continue
    if flag==1:
        continue
    if competitive_comp == entry.composition:
        continue
    num_atoms = get_num_atoms(str(competitive_comp).replace(' ', ''))
    #print(num_atoms)
    competitive_energy = comp['energy_per_atom'] * num_atoms

    competitive_entry = ComputedEntry(
        composition=competitive_comp, energy=competitive_energy)
    competitive_entries.append(competitive_entry)
all_entries = [entry] + competitive_entries


#dumpfn(competitive_entries, "./competitive_entries.json") # or .json.gz


#Process entries with Materials Project compatibility.
#all_entries = compat.process_entries(all_entries)

pd = PhaseDiagram(all_entries)
decomp, energy_above_hull = pd.get_decomp_and_e_above_hull(
    entry)

with open("./ehull.txt","w") as fn:
    fn.write(str(space_group_number)+','+str(round(energy_above_hull,5))+','+str(round(entry.energy_per_atom,5))) 
