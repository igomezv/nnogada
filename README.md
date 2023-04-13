[<img src="https://img.shields.io/badge/astro--ph.IM-%20%09arXiv%3A2209.02685-green.svg">](https://arxiv.org/abs/2209.02685)

# Neural Networks Optimized by Genetic Algorithms for Data Analysis (NNOGADA) 

**nnogada** is a Python package that performs hyperparemeter tuning for artificial neural networks, particularly for Multi Layer Perceptrons, using simple genetic algorithms. Useful for generate better neural network models for data analysis. Currently, only works with feedforward neural networks in tensorflow.keras (classification and regression) and torch (regression at this moment).

Before use the code, please install the requirements:

    $ pip3 install -r requiriments.txt
 
Or you can try to install nnogada in your computer:

     $ git clone https://github.com/igomezv/nnogada

     $ cd nnogada

     $ pip3 install -e .

then you can delete the cloned repo because you must have nnogada installed locally.

Other way to install nnogada (without clonning) is:

    $ pip3 install -e git+https://github.com/igomezv/nnogada#egg=nnogada


If you use the code, please cite the paper *Gómez-Vargas, I., Andrade, J. B., & Vázquez, J. A. (2023). Neural networks optimized by genetic algorithms in cosmology. Physical Review D, 107(4), 043509.*

Contributions are welcome!

![](https://raw.githubusercontent.com/igomezv/igomezv.github.io/master/assets/img/nnogada_output.png)

## Description of this repository

- nnogada folder has the source code of the neural approximator with genetic algorithms.
- In data folder we have some data used in tests. 
- example_1.py is a classification task of galaxies, quasars and stars from SDSS using a keras neural model.
- example_2.py is a regression task with SNeIa using a keras neural model.
- example_3.py is the same as example_2.py but with a torch neural model.

## TODO 

- Include convolutional, recurrent neural networks and other architectures.
- To allow pytorch models for classification.