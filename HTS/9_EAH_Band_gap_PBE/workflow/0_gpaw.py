# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import numpy as np
from ase.build import bulk
from ase.optimize.bfgs import BFGS
from ase.constraints import UnitCellFilter
from ase.constraints import StrainFilter
from gpaw import GPAW
from gpaw import PW
from ase.io import read, write
si = read('temp.cif')
#si = bulk('Si', 'fcc', a=6.0)
# Experimental Lattice constant is a=5.421 A

si.calc = GPAW(xc='PBE',
               mode=PW(340),
               kpts={'density': 2, 'gamma': True},
               #convergence={'eigenstates': 1.e-10},  # converge tightly!
               txt='stress.txt')

sf = UnitCellFilter(si)
opt = BFGS(sf)
opt.run(0.2)

write('temp.cif', si)

