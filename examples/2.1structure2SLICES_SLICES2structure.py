from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
# obtaining the pymatgen Structure instance of NdSiRu
original_structure = Structure.from_file(filename='NdSiRu.cif')
# creating an instance of the InvCryRep Class (initialization)
backend=InvCryRep()
# converting a crystal structure to its SLICES string
slices_NdSiRu=backend.structure2SLICES(original_structure) 
# converting a SLICES string back to its original crystal structure and obtaining its M3GNet_IAP-predicted energy_per_atom
reconstructed_structure,final_energy_per_atom_IAP = backend.SLICES2structure(slices_NdSiRu)
print('SLICES string of NdSiRu is: ',slices_NdSiRu)
print('\nReconstructed_structure is: ',reconstructed_structure)
print('\nfinal_energy_per_atom_IAP is: ',final_energy_per_atom_IAP,' eV/atom')
# if final_energy_per_atom_IAP is 0, it means the M3GNet_IAP refinement failed, and the reconstructed_structure is the ZL*-optimized structure.

