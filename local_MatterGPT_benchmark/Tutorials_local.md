# Local Installation
```bash
# For users in China, configure pip for a faster mirror:
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install environment
conda env create --name slices --file=environments.yml
conda activate slices
pip install slices
```
Please note that this installtion method is intended for Linux operating systems like Ubuntu or Ubuntu on win11. To run SLICES on MacOS, one can run SLICES with docker, referring to [Jupyter backend setup](#jupyter-backend-setup).

If "TypeError: bases must be types" occurs when you use SLICES library, then do this:
```bash
pip install protobuf==3.20.0
```

# Run MatterGPT locally
## Tutorial 2.1 Inverse Design of Novel Materials with targeted formation energy

```bash
# Build training sets
cd local_MatterGPT_benchmark/mp20_nonmetal
python 1_run.py

# Train MatterGPT for Single-Property Material Inverse Design (using formation energy as an example)
cd ../eform/1_train_generate
sh 1_train.sh

# Generate SLICES strings with specified $E_{form}$ = [-1,-2,-3,-4]  eV/atom
sh 2_generate_canon_SLICES.sh

# Reconstruct crystals from SLICES, assess novelty, predict formation energy and visualize
cd ../2_inverse_eform_novelty
python 1_run.py

# Evaluate the formation energy distribution of the reconstructed crystals at PBE level (need workstation or even HPC to run VASP fastly)
# Need to modify the scripts in workflow
# Need VASP on your workstation or HPC
cd ../3_eform_DFT
python 1_run.py
```
## Tutorial 2.2 Inverse Design of Novel Materials with targeted bandgap
```bash
# Train MatterGPT for Single-Property Material Inverse Design
cd local_MatterGPT_benchmark/bandgap/1_train_generate
sh 1_train.sh

# Generate SLICES strings with specified $E_{gap}$ = [1,2,3,4]  eV/atom
sh 2_generate_canon_SLICES.sh

# # Reconstruct crystals from SLICES and assess novelty
cd ../2_inverse_novelty
python 1_run.py

# Evaluate the bandgap distribution of the reconstructed crystals at PBE level (need workstation or even HPC to run VASP fastly)
# Need to modify the scripts in workflow
# Need VASP on your workstation or HPC
cd ../3_DFT
python 1_run.py
```

## Tutorial 2.3 Inverse Design of Novel Materials with both targeted bandgap and targeted formation energy
```bash
# Train MatterGPT
cd local_MatterGPT_benchmark/bandgap_eform/1_train_generate
sh 1_train.sh

# Generate SLICES strings with specified [$E_{form}$, $E_{gap}$] = [-2.0 eV/atom, 1.0 eV]  
sh 2_generate_canon_SLICES.sh

# # Reconstruct crystals from SLICES and assess novelty
cd ../2_inverse_novelty
python 1_run.py

# Evaluate the bandgap & eform distribution of the reconstructed crystals at PBE level (need workstation or even HPC to run VASP fastly)
# Need to modify the scripts in workflow
# Need VASP on your workstation or HPC
cd ../3_DFT
python 1_run.py
```

## Benchmark of SLI2Cry
```bash
# Build filtered MP-20 dataset
cd local_MatterGPT_benchmark/mp20
python 1_run.py

# Run Benchmark
cd ../1_Match_rate_MP-20
python 1_run.py
```