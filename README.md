# Simplified Line-Input Crystal-Encoding System (SLICES)

The **Simplified Line-Input Crystal-Encoding System (SLICES)** is the first invertible and invariant crystal representation tool. This software supports encoding and decoding crystal structures, reconstructing them, and generating new materials with desired properties using generative deep learning.

**Related Publications and Resources:**
- **Nature Communications**: [Paper](https://www.nature.com/articles/s41467-023-42870-7)
- **SLICES Video Introduction**: [Bilibili](https://www.bilibili.com/video/BV17H4y1W7aZ/)
- **SLICES101**: [Bilibili](https://www.bilibili.com/video/BV1Yr42147dM/)
- **Data and Results**: [Figshare](https://doi.org/10.6084/m9.figshare.22707472)
- **MatterGPT Paper**: [arXiv](https://arxiv.org/abs/2408.07608)
- **MatterGPT Demo**: [Huggingface](https://huggingface.co/spaces/xiaohang07/MatterGPT_CPU)
- **SLICES-PLUS Paper**: [arXiv](https://arxiv.org/abs/2410.22828)
---

## Main Functionalities

1. **Encode crystal structures into SLICES strings**
2. **Reconstruct original crystal structures (Text2Crystal)**
3. **Inverse design of solid-state materials with desired properties using MatterGPT**
4. **Inverse design of solid-state materials with desired properties and crystal systems using MatterGPT ([SLICES-PLUS](https://arxiv.org/abs/2410.22828))**

---
We provide a huggingface space to allow one-click conversion of CIF to SLICES and SLICES to CIF online. 
### [[Online SLICES/CIF Convertor]](https://huggingface.co/spaces/xiaohang07/SLICES)
[![IMAGE ALT TEXT](./docs/SLICES_demo.png)](https://huggingface.co/spaces/xiaohang07/SLICES "Online SLICES/CIF Convertor - Click to Try!")
### [[MatterGPT Online Demo]](https://huggingface.co/spaces/xiaohang07/MatterGPT_CPU)
[![IMAGE ALT TEXT](./docs/huggingface_space.png)](https://huggingface.co/spaces/xiaohang07/MatterGPT_CPU "MatterGPT Online Demo - Click to Try!")

---

## Table of Contents

1. [Installation](#installation)
   - [Local Installation](#local-installation)
   - [Docker Installation for Jupyter Backend](#docker-installation-for-jupyter-backend)
2. [Examples](#examples)
   - [Crystal to SLICES and SLICES to Crystal](#crystal-to-slices-and-slices-to-crystal)
   - [Augment SLICES and Canonicalize SLICES](#augment-slices-and-canonicalize-slices)
3. [Tutorials](#tutorials)
4. [Documentation](#documentation)
5. [Reproducing Benchmarks](#reproducing-benchmarks)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgement)
8. [Contact and Support](#contact-and-support)

---

## Installation

### Local Installation
The local version does not require Docker. Follow the steps below to set up the environment:

```bash
# For users in China, configure pip for a faster mirror:
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install --upgrade pip

# Install environment
conda env create --name slices --file=environments.yml
conda activate slices
pip install slices
```

**Important Notes:**
- SLICES works on **Linux systems** like Ubuntu or WSL2 (Windows Subsystem for Linux).
- It is not directly compatible with Windows or MacOS due to dependency on a modified XTB binary.
- For Windows/MacOS users, use Docker instead ([Docker Setup](#docker-installation-for-jupyter-backend)).

**Troubleshooting:**
- If you encounter `TypeError: bases must be types`, fix it by running:

```bash
pip install protobuf==3.20.0
```
- If errors persist, consider using Docker instead.
### Docker Installation for Jupyter Backend
Follow these steps to set up SLICES using Docker:

1. **Download the Repository** and unzip it.
2. **Insert Materials Project API Key** into `APIKEY.ini`.
3. **Configure CPU threads** in `slurm.conf`.
4. **Run Docker Setup Commands**:

```bash
# Pull the prebuilt SLICES Docker image
docker pull xiaohang07/slices:v9

# you can build your own docker image using the Dockerfile in this repo. Many thanks to Prof. Haidi Wang (https://haidi-ustc.github.io/about/) for the Dockerfile.
# You can download the compressed docker image v9 at https://figshare.com/s/260701a1accd0192de20 if docker pull does not work. 
# Then you can load this docker image using the following command: 
xz -dc slices_v9.tar.xz | docker load

# Make scripts executable
sudo chmod +x entrypoint_set_cpus_jupyter.sh ./slices/xtb_noring_nooutput_nostdout_noCN

# Run Docker (replace [] with your absolute path)
docker run -it -p 8888:8888 -h workq --shm-size=0.5gb --gpus all -v /[]:/crystal xiaohang07/slices:v9 /crystal/entrypoint_set_cpus_jupyter.sh
```

5. **Access Jupyter Notebook**:
   - Press `CTRL` (or `Command` on Mac) and click the `http://127.0.0.1` link in the terminal.
   - Open the relevant tutorial notebook, e.g., `Tutorial_*.ipynb`.

**Best Practice:**
Please note: The best way to run this project is using Docker on Windows 11, as this allows you to utilize GPU for model training directly within Docker containers. In contrast, when running Docker on Ubuntu, accessing GPU from within Docker containers has proven problematic (verified across multiple machines). Therefore, on Ubuntu systems, a hybrid approach is required: install PyTorch directly on the host machine for training MatterGPT models and generating SLICES, while other steps can be run in Docker. 
请注意：在所有操作系统中，运行此项目的最佳方式是使用 Windows 11 的 Docker 环境，因为它能够让你在 Docker 容器内直接调用 GPU 来训练模型。相比之下，在 Ubuntu 系统中运行 Docker 时会遇到 GPU 调用的问题（经过多台计算机测试验证），因此在 Ubuntu 上需要采用混合方案：在本机直接安装 PyTorch 来训练 MatterGPT 模型和生成 SLICES，而其他步骤则可以在 Docker 容器中完成。
- **Windows 11**: Use Docker with GPU support.
- **Ubuntu**: Hybrid setup: Install PyTorch locally for model training; use Docker for other steps.

---

## Examples

### Crystal to SLICES and SLICES to Crystal
Convert a crystal structure to its SLICES string and reconstruct it.

```python
from slices.core import SLICES
from pymatgen.core.structure import Structure

# Load crystal structure from file
original_structure = Structure.from_file(filename='NdSiRu.cif')
backend = SLICES()

# Convert to SLICES string
slices_NdSiRu = backend.structure2SLICES(original_structure)

# Reconstruct crystal and get predicted energy
reconstructed_structure, final_energy_per_atom = backend.SLICES2structure(slices_NdSiRu)

print('SLICES string of NdSiRu is:', slices_NdSiRu)
print('Reconstructed structure:', reconstructed_structure)
print('Final energy per atom:', final_energy_per_atom, 'eV/atom')
```

### Augment SLICES and Canonicalize SLICES
Generate augmented SLICES strings and reduce them to a canonical form.

```python
from slices.core import SLICES
from pymatgen.core.structure import Structure

# Load crystal structure
original_structure = Structure.from_file(filename='Sr3Ru2O7.cif')
backend = SLICES(graph_method='econnn')

# Generate augmented SLICES
slices_list = backend.structure2SLICESAug_atom_order(structure=original_structure, num=50)
canonical_slices = list(set(backend.get_canonical_SLICES(s) for s in slices_list))

print('Unique Canonical SLICES:', len(canonical_slices))
```

---
## Tutorials
### Tutorials for Docker Installation
- **SLICES Video Tutorials**: [Bilibili](https://space.bilibili.com/398676911/channel/seriesdetail?sid=4012344)
- **MatterGPT Video Tutorials**: [Bilibili](https://www.bilibili.com/video/BV1agsLeUEAB)
- **Jupyter Setup Instructions**: See [Docker Setup](#docker-installation-for-jupyter-backend)
- **Jupyter Examples**:
   - [Introductory Examples](./Tutorial_1.0_Intro_Example.ipynb)
   - [Single-Property Material Design (eform)](./Tutorial_2.1_MatterGPT_eform.ipynb)
   - [Single-Property Material Design (bandgap)](./Tutorial_2.2_MatterGPT_bandgap.ipynb)
   - [Multi-Property Design](./Tutorial_2.3_MatterGPT_2props_bandgap_eform.ipynb)
### [Tutorials for Local Installation](./local_MatterGPT_benchmark/Tutorials_local.md) 
---

## Documentation
The official documentation is available at [Read the Docs](https://xiaohang007.github.io/SLICES/).

---

## Reproducing Benchmarks
Refer to the [Benchmarks Guide](benchmark/benchmarks.md) for detailed instructions.

---

## Citation

If you use SLICES, MatterGPT or SLICES-PLUS, please cite the following works:

```bibtex
@article{xiao2023invertible,
  title={An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning},
  author={Xiao, Hang and Li, Rong and Shi, Xiaoyang and Chen, Yan and Zhu, Liangliang and Chen, Xi and Wang, Lei},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7027},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@misc{chen2024mattergptgenerativetransformermultiproperty,
  title={MatterGPT: A Generative Transformer for Multi-Property Inverse Design of Solid-State Materials},
  author={Yan Chen and Xueru Wang and Xiaobin Deng and Yilun Liu and Xi Chen and Yunwei Zhang and Lei Wang and Hang Xiao},
  year={2024},
  eprint={2408.07608},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci},
  url={https://arxiv.org/abs/2408.07608}
}

@misc{wang2024slicespluscrystalrepresentationleveraging,
      title={SLICES-PLUS: A Crystal Representation Leveraging Spatial Symmetry}, 
      author={Baoning Wang and Zhiyuan Xu and Zhiyu Han and Qiwen Nie and Hang Xiao and Gang Yan},
      year={2024},
      eprint={2410.22828},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2410.22828}, 
}
```

---

## Acknowledgement
Special thanks to the open-source projects and developers that inspired this work:
- [tobascco](https://github.com/peteboyd/tobascco)
- [xtb](https://github.com/grimme-lab/xtb)
- [m3gnet](https://github.com/materialsvirtuallab/m3gnet)
- [chgnet](https://github.com/CederGroupHub/chgnet)
- [molgpt](https://github.com/devalab/molgpt)

---

## Contact and Support
- **Email**: [hangxiao@ln.edu.hk](mailto:hangxiao@ln.edu.hk)
- **ResearchGate**: [Hang Xiao](https://www.researchgate.net/profile/Hang-Xiao-8)
- **Start a Discussion**: [GitHub Discussions](https://github.com/xiaohang007/SLICES/discussions/categories/general)

