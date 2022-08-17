import time
from nnogada.Nnogada import Nnogada
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


df = pd.read_csv('data/jla.csv')
N = len(df.values)
randomize = np.random.permutation(N)
data = df.values[randomize]
N = len(df.values)
z = data[:, 0]
y = data[:, 1:3] ### toma el resto de variables a predecir
y[:,1] = y[:, 1]**2+data[:,2]
np.shape(y)


dmag = df["dmb"]
df2 = df['errors']+df['dmb']**2

scalerz = StandardScaler()
scalerz.fit(z.reshape(-1, 1))
z = scalerz.transform(z.reshape(-1, 1))

split = 0.75
ntrain = int(split * len(z))
indx = [ntrain]
X_train, X_val = np.split(z, indx)
Y_train, Y_val = np.split(y, indx)

population_size = 11   # max of individuals per generation
max_generations = 5    # number of generations
gene_length = 4        # lenght of the gene, depends on how many hiperparameters are tested
k = 1                  # num. of finalist individuals

t = time.time()
datos = []

# Define the hyperparameters for the search
hyperparams = {'deep': [1, 2], 'num_units': [1,2], 'batch_size': [256,1048]}

# generate a Nnogada instance
net_fit = Nnogada(hyp_to_find=hyperparams, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
# Set the possible values of hyperparameters and not use the default values from hyperparameters.py
net_fit.set_hyperparameters()

# best solution
best_population = net_fit.ga_with_elitism(population_size, max_generations, gene_length, k)
print(best_population)
print("Total elapsed time:", (time.time()-t)/60, "minutes")

