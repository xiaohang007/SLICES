# -*- coding: utf-8 -*-
# created by dagleaves https://github.com/dagleaves/ehull-calculator
# modified by Hang Xiao 2023.04
# xiaohang007@gmail.com
import os
import re
import csv
import sys
import time
import argparse
import pickle
from monty.serialization import loadfn, dumpfn


from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core.composition import Composition



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



def get_E_above_hull_load(formula_list,E_atom_list):
    entry_list=[]
    elements_list=[]
    energy_above_hull_list=[]
    expanded_formula_list=[]
    for i in range(len(formula_list)):
        formula = formulaExpander(formula_list[i])
        expanded_formula_list.append(formula)
        num_atoms = get_num_atoms(formula)
        E_total = E_atom_list[i] * num_atoms
        comp = Composition(formula)
        entry_list.append(ComputedEntry(composition=comp, energy=E_total)) 
    energy_above_hull_list=[]
    competitive_compositions=loadfn("../competitive_compositions.json.gz")
    for i in range(len(entry_list)):
        competitive_entries = []
        for comp in competitive_compositions:
            flag=0
            competitive_comp = comp['composition']  # cannot use comp.composition like native MP object.
            for j in competitive_comp.elements:
                if j not in Composition(expanded_formula_list[i]).elements:
                    flag=1
                    continue
            if flag==1:
                continue
            if str(competitive_comp).replace(' ', '') == expanded_formula_list[i]:
                continue
            num_atoms = get_num_atoms(str(competitive_comp).replace(' ', ''))
            #print(num_atoms)
            competitive_energy = comp['energy_per_atom'] * num_atoms

            competitive_entry = ComputedEntry(
                composition=competitive_comp, energy=competitive_energy)
            competitive_entries.append(competitive_entry)
        all_entries = [entry_list[i]] + competitive_entries
        phase_diagram = PhaseDiagram(all_entries)
        decomp, energy_above_hull = phase_diagram.get_decomp_and_e_above_hull(
            entry_list[i])
        energy_above_hull_list.append(energy_above_hull)
        with open('temp1','a') as f:
            f.write(str(i)+'\n')
    return energy_above_hull_list



def main(args):
    count=0
    save = []
    formula_list=[]
    E_atom_list=[]
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            formula_list.append(row[3])
            E_atom_list.append(float(row[5]))

    print(1)

    energy_above_hull_list=get_E_above_hull_load(formula_list,E_atom_list)
    print(len(formula_list),len(energy_above_hull_list))
    print(formula_list[0],energy_above_hull_list[0],formula_list[-1],energy_above_hull_list[-1])
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            with open(args.output, 'a', newline='') as f:
                writer = csv.writer(f)
                row.append(energy_above_hull_list[count])
                writer.writerow(row)
                count+=1

"""
                decomp, energy_above_hull = get_E_above_hull(formula, E_atom)

                with open(args.output, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],energy_above_hull])
            except:
                pass
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculates energy above the convex hull for a csv of formulas and total energies (with header)')
    parser.add_argument('--debug', action='store_false',
                        help='Enable debug mode (test imports)')
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input csv filename')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='Output csv filename')
    args = parser.parse_args(sys.argv[1:])

    assert os.path.exists(args.input), 'Input file does not exist'

    main(args)
