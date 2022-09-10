#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup


with open("README.md", "r") as fh:
    desc = fh.read()

required = ["numpy", "scipy", "sklearn", "tensorflow", "deap", 'bitstring']

setup(
    name="nnogada",
    version='0.0.1',    
    author='I Gomez-Vargas',
    author_email="igomez@icf.unam.mx",
    url="https://github.com/igomezv/nnogada",
    license="MIT",
    description="Genetic hyperparameter tuning for neural nets",
    long_description=desc,
    install_requires=required,
    include_package_data=False,
    keywords=["Hyperparameter",
              "optimization",
              "machine learning",
              "deep learning",
              "genetic algorithms"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Natural Language :: English',
    ],

)
