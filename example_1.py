import time
from neurapprox.GeneticAlgorithm import GeneticAlgorithm

population_size = 11   # max of individuals per generation
max_generations = 5    # number of generations
gene_length = 4       # lenght of the gene, depends on how many hiperparameters are tested
k = 1;                 # num. of finalist individuals

t = time.time()
datos = []

neurapprox = GeneticAlgorithm()
ss = [i for i in range(1,population_size*(max_generations+1))]
best_population = geneticAlgorithm_with_elitism(population_size, max_generations, gene_length, k)
print("Total elapsed time:", (time.time()-t)/60, "minutes")

