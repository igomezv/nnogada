# This is a classification example with keras
# At this moment, only with keras we can perform classification tasks.
import time
from nnogada.Nnogada import Nnogada
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as split
import numpy as np
import requests

# Divide data into X and Y and implement hot_ones in Y
def prepare_dataset(data):
    X, Y = np.empty((0)), np.empty((0))
    X = data[:,0:8]
    Y = data[:,8]
    Y = to_categorical(Y, num_classes=3)
    return X, Y

url = "https://raw.githubusercontent.com/igomezv/nnogada/main/data/star_classification.csv"
data = pd.read_csv(url)
cols = ['alpha','delta','u','g','r','i','z','redshift','class']
data = data[cols]

data["class"] = [0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in data["class"]]
print(data.head())
data = data.to_numpy()

# Split dataset into train, validation and test sets
X,Y = prepare_dataset(data)

# Defines ratios, w.r.t. whole dataset.
ratio_train = 0.8
ratio_val = 0.1
ratio_test = 0.1

# Produces test split.
x_, X_test, y_, Y_test = split(X, Y, test_size = ratio_test, random_state=0)

# Adjusts val ratio, w.r.t. remaining dataset.
ratio_remaining = 1 - ratio_test
ratio_val_adjusted = ratio_val / ratio_remaining

# Produces train and val splits.
X_train, X_val, Y_train, Y_val = split(x_, y_, test_size=ratio_val_adjusted, random_state=0)

scaler = StandardScaler()
# Normalize and scale the input sets.
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
X_val   = scaler.transform(X_val)

population_size = 11   # max of individuals per generation
max_generations = 5    # number of generations
gene_length = 8        # lenght of the gene, depends on how many hiperparameters are tested 2*hyp
k = 1                  # num. of finalist individuals

t = time.time()
datos = []

# Define the hyperparameters for the search
hyperparams = {'deep': [1,2], 'num_units': [1,5], 'batch_size': [128, 256]}

# generate a Nnogada instance
net_fit = Nnogada(hyp_to_find=hyperparams, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, regression=False)
# Set the possible values of hyperparameters and not use the default values from hyperparameters.py
net_fit.set_hyperparameters()

# best solution
best_population = net_fit.ga_with_elitism(population_size, max_generations, gene_length, k)
print(best_population)
print("Total elapsed time:", (time.time()-t)/60, "minutes")

