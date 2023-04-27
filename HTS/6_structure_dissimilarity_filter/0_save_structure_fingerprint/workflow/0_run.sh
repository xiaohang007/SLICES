#/bin/bash

rm result.csv
python gen_structure_fingerprint.py -i "temp.json" -o result.csv
