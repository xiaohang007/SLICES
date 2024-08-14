from slices.core import SLICES
from pymatgen.core.structure import Structure
# obtaining the pymatgen Structure instance of NdSiRu

# creating an instance of the InvCryRep Class (initialization)
backend=SLICES(relax_model="chgnet")
#backend=SLICES(relax_model="m3gnet")
# converting a crystal structure to its SLICES string
slices="Sb Ti Ti Ir 0 2 oo- 0 2 ooo 0 2 o-o 0 2 -oo 0 3 ooo 0 3 o-o 0 3 oo- 0 3 -o- 1 2 ooo 1 2 oo- 1 2 o-- 1 2 o-o 1 3 ooo 1 3 o-- 1 3 o-o 1 3 -o- "
# converting a SLICES string back to its original crystal structure and obtaining its M3GNet_IAP-predicted energy_per_atom
reconstructed_structure,final_energy_per_atom_IAP = backend.SLICES2structure(slices)


print('\nReconstructed_structure is: ',reconstructed_structure)
print('\nfinal_energy_per_atom_IAP is: ',final_energy_per_atom_IAP,' eV/atom')
reconstructed_structure.to("1.cif")
# if final_energy_per_atom_IAP is 0, it means the M3GNet_IAP refinement failed, and the reconstructed_structure is the ZL*-optimized structure.

