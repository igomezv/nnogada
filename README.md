# Neural Networks Optimized by Genetic Algorithms for Data Analysis (NNOGADA)

[![PyPI version](https://badge.fury.io/py/nnogada.svg)](https://badge.fury.io/py/nnogada)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![arXiv](https://img.shields.io/badge/arXiv-2209.02685-b31b1b.svg)](https://doi.org/10.48550/arXiv.2209.02685)
[![GitHub Repo stars](https://img.shields.io/github/stars/igomezv/nnogada?style=social)](https://github.com/igomezv/nnogada)

**NNOGADA** is a Python package designed to optimize hyperparameters for artificial neural networks, specifically for Multi-Layer Perceptrons (MLP), through simple genetic algorithms. It is particularly effective in generating improved neural network models for data analysis, with current support for feedforward neural networks in tensorflow.keras (for both classification and regression) and torch (currently supporting regression).

## Installation

You can install **NNOGADA** directly from the source or via pip. To install from the source:

```bash
git clone https://github.com/igomezv/nnogada
cd nnogada
pip3 install -e .
```

After installation, you may remove the cloned repository as **NNOGADA** will be installed locally.

Alternatively, to install **NNOGADA** without cloning the repository:

```bash
pip3 install -e git+https://github.com/igomezv/nnogada#egg=nnogada
```
**Note:** Torch performance has been observed to be more efficient on laptop GPUs running Linux.

## Usage

**NNOGADA** provides a simple interface for optimizing neural network models. Examples included in this repository (`example_1.py`, `example_2.py`, `example_torch.py`) offer a good starting point for understanding how to utilize the package for your data analysis tasks.

## Citing NNOGADA

If you find **NNOGADA** useful in your research, please consider citing our paper:

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
  doi={10.48550/arXiv.2209.02685},
  url={https://doi.org/10.48550/arXiv.2209.02685}
}
```


## Contributions

Contributions to **NNOGADA** are very welcome! If you have suggestions for improvements or new features, feel free to create an issue or pull request.

## TODO

- [ ] Include support for other neural network architectures such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

## Project Structure

- **data/**: Sample data files.
- **docs/**: Documentation files. Use `docs_sphinx` for the latest Sphinx documentation.
- **nnogada/**: Core library files.
- **outputs/**: Model output and logs.
- **examples/**: Example scripts to demonstrate library usage.
- **requirements.txt**: List of package dependencies.
- **setup.py**, **setup.cfg**: Packaging and installation scripts.

## License

NNOGADA is licensed under the GPL v2 license. See the [LICENSE](LICENSE) file for more details.

