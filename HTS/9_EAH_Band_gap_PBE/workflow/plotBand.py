# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter

vaspout = Vasprun("./vasprun.xml")
bandstr = vaspout.get_band_structure(kpoints_filename="KPOINTS",line_mode=True)
plt = BSPlotter(bandstr).get_plot(ylim=[-2.5,2.5])
plt.savefig("band.png")
