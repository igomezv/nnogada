Neural Networks Optimized by Genetic Algorithms for Data Analysis (NNOGADA) 

INFO:

**nnogada** is a Python package that performs hyperparemeter tuning for artificial neural networks, particularly for Multi Layer Perceptrons, using simple genetic algorithms. Useful for generate better neural network models for data analysis. Currently, only works with feedforward neural networks in tensorflow.keras (classification and regression) and torch (regression at this moment).

You can try to install nnogada in your computer:

     $ git clone https://github.com/igomezv/nnogada

     $ cd nnogada

     $ pip3 install -e .

then you can delete the cloned repo because you must have nnogada installed locally.

Other way to install nnogada (without clonning) is:

    $ pip3 install -e git+https://github.com/igomezv/nnogada#egg=nnogada

If you use the code, please cite the paper *Gómez-Vargas, I., Andrade, J. B., & Vázquez, J. A. (2023). Neural networks optimized by genetic algorithms in cosmology. Physical Review D, 107(4), 043509.*

**Note**: Torch seems to have better performance on a laptop GPU in linux.

Contributions are welcome!

TODO:
- To include other architectures such as conv and reccurrent nets.
