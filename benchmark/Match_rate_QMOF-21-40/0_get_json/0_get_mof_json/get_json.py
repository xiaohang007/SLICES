import pandas as pd
from pymatgen.io.cif import CifParser
import json
import os


data = pd.read_csv('qmof.csv')


selected_data = data[(data['info.natoms'] > 20) & (data['info.natoms'] <= 40)]


data_list = []


for _, row in selected_data.iterrows():
    qmof_id = row['qmof_id']

    structure = CifParser(os.path.join('relaxed_structures', f'{qmof_id}.cif')).get_structures()[0]

    cif_string = str(structure.to(fmt="cif"))

    data_list.append({'qmof_id': qmof_id, 'cif': cif_string})


with open('cifs.json', 'w') as f:
    json.dump(data_list, f)
