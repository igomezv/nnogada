---
title: 'nnogada: neural networks optimized by genetic algorithms for data analysis'
tags:
  - Python
  - cosmology
  - regression
  - neural networks
authors:
  - name: Isidro Gómez-Vargas
    orcid: 0000-0002-6473-018X
    equal-contrib: true
    corresponding: true #
    affiliation: 1
  - name: José Alberto Vázquez
    affiliation: 1
affiliations:
 - name: Instituto de Ciencias Físicas, Universidad Nacional Autónoma de México
   orcid: 0000-0002-7401-0864
   index: 1

date: 18 April 2023
bibliography: paper.bib

---

# Summary

Neural networks can be used to address regression or classification problems in cosmology, but achieving high precision is essential, particularly in regression scenarios where well-tuned neural networks are necessary. To improve the efficiency of finding optimal hyperparameters, genetic algorithms are used instead of traditional grid methods. `nnogada` is a Python-based software that simplifies hyperparameter tuning for regression problems by utilizing genetic algorithms within popular deep learning libraries.


# Statement of need

Selecting the appropriate hyperparameters for a neural network is crucial as it ensures an accurate model without underfitting or overfitting the training data. Several strategies have been proposed to identify suitable values for the hyperparameters in a neural network [@larochelle:2007; @hutter:2009; @bardenet:2013; @zhang:2019]. The standard approach involves creating a multidimensional grid that specifies several values for the hyperparameters [@larochelle:2007]. Then, all possible combinations of the hyperparameters are evaluated, and the combination that exhibits the best performance is selected through comparison. In recent years, alternative methods have emerged that rely on mathematical optimization or metaheuristic algorithms, which employ specialized techniques to search for the optimal value of a given function. Among these, genetic algorithms have gained attention due to their efficiency in searching for the best combination of hyperparameters.

`nnogada` is a Python package that employs genetic algorithms, implemented within the `deap` Python library [@de:2012], to search for hyperparameters of feedforward neural networks. Originally designed as a tool for cosmology, it was created to develop highly accurate models for training purposes. A comparison with traditional methods in cosmological applications is discussed in [@Gomez:2023] to demonstrate its advantages. However, this package can be utilized in any field that requires feedforward neural networks, regardless of the nature of the dataset. It is compatible with both keras and torch neural network models.

`nnogada` comprises of two primary classes: the `Nnogada` class, which creates and trains neural networks (using `torch` or `keras`) while implementing genetic algorithms to determine the optimal architecture, and the `Hyperparameter` class, which enables the definition of hyperparameters and their search space, as well as the ability to specify whether each parameter is fixed or included in the genetic algorithm search.

In conclusion, `nnogada` has demonstrated its effectiveness in regression and classification results in cosmological applications, particularly in model-independent reconstructions and learning cosmological functions. Further investigation is needed to fully explore its capabilities in this domain.


# Acknowledgements

JAV acknowledges the support provided by FOSEC SEP-CONACYT Investigación Básica A1-S-21925, Ciencias de Frontera CONACYT-PRONACES/304001/202 and UNAM-DGAPA-PAPIIT IA104221. IGV thanks the CONACYT postdoctoral fellowship and the support of the ICF-UNAM.

# References
