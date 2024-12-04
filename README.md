# Neural Networks Optimized by Genetic Algorithms for Data Analysis (`NNOGADA`)

[![PyPI version](https://badge.fury.io/py/nnogada.svg)](https://badge.fury.io/py/nnogada)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![arXiv](https://img.shields.io/badge/arXiv-2209.02685-b31b1b.svg)](https://doi.org/10.48550/arXiv.2209.02685)
[![GitHub Repo stars](https://img.shields.io/github/stars/igomezv/nnogada?style=social)](https://github.com/igomezv/nnogada)

**`NNOGADA`** is a Python package designed to optimize hyperparameters for artificial neural networks, specifically for Multi-Layer Perceptrons (MLP), through simple genetic algorithms. It is particularly effective in generating improved neural network models for data analysis, with current support for feedforward neural networks in tensorflow.keras (for both classification and regression) and torch (currently supporting regression).

## Documentation

For an introduction and API documentation, visit [`nnogada` Docs](https://igomezv.github.io/nnogada).

## Installation

You can install **`nnogada`** directly from the source or via pip. To install from the source:

```bash
git clone https://github.com/igomezv/nnogada
cd nnogada
pip3 install -e .
```

After installation, you may remove the cloned repository as **`nnogada`** will be installed locally.

Alternatively, to install **`nnogada`** without cloning the repository:

```bash
pip3 install -e git+https://github.com/igomezv/nnogada#egg=nnogada
```
**Note:** Torch performance has been observed to be more efficient on laptop GPUs running Linux.

## Usage

**NNOGADA** provides a simple interface for optimizing neural network models. Examples included in this repository (`example_1.py`, `example_2.py`, `example_torch.py`) offer a good starting point for understanding how to utilize the package for your data analysis tasks.

## Citing `nnogada`

If you find **`nnogada`** useful in your research, please consider citing [our paper](https://arxiv.org/abs/2209.02685):

```bibtex
@article{GomezVargas2023,
  title={Neural networks optimized by genetic algorithms in cosmology},
  author={Gómez-Vargas, I. and Andrade, J. B. and Vázquez, J. A.},
  journal={Physical Review D},
  volume={107},
  number={4},
  pages={043509},
  year={2023},
  publisher={American Physical Society},
  doi={https://doi.org/10.1103/PhysRevD.107.043509},
  url={https://doi.org/10.48550/arXiv.2209.02685}
}
```


## Contributions

Contributions to **`nnogada`** are very welcome! If you have suggestions for improvements or new features, feel free to create an issue or pull request.

## TODO

- [ ] To include support for other neural network architectures such as Convolutional Neural Networks (CNNs) and Variational Autoencoders (VAEs).

## Project Structure

- **data/**: Sample data files.
- **docs/**: Documentation files. Use `docs_sphinx` for the latest Sphinx documentation.
- **nnogada/**: Core library files.
- **outputs/**: Model output and logs.
- **examples files**: Example scripts to demonstrate library usage.
- **requirements.txt**: List of package dependencies.
- **setup.py**, **setup.cfg**: Packaging and installation scripts.

## License

NNOGADA is licensed under the MIT license. See the [LICENSE](LICENSE) file for more details.

## Research works that use **`nnogada`**:

- Mitra, A., Gómez-Vargas, I., & Zarikas, V. (2024). Dark energy reconstruction analysis with artificial neural networks: Application on simulated Supernova Ia data from Rubin Observatory. Physics of the Dark Universe, 46, 101706.
- Gómez-Vargas, I., & Vázquez, J. A. (2024). Deep Learning and genetic algorithms for cosmological Bayesian inference speed-up. Physical Review D, 110(8), 083518.
- Garcia-Arroyo, G., Gómez-Vargas, I., & Vázquez, J. A. (2024). Reconstructing rotation curves with artificial neural networks. arXiv preprint arXiv:2404.05833.

