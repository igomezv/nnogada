# import deap
from deap import base, creator, tools, algorithms
from bitstring import BitArray
from elitism import eaSimpleWithElitism
import numpy as np
from scipy.stats import bernoulli

class GeneticAlgorithm:
    def __init__(self, hyp_list, model, train_evaluate):
        self.hyp_list = hyp_list
        self.model = model
        self.train_evaluate = train_evaluate

    def ga_with_elitism(self, population_size, max_generations, gene_length, k):
        # Genetic Algorithm constants:
        P_CROSSOVER = 0.5  # probability for crossover
        P_MUTATION = 0.5  # probability for mutating an individual
        HALL_OF_FAME_SIZE = 1  # Best individuals that pass to the other generation

        # set the random seed:
        toolbox = base.Toolbox()

        # As we are trying to minimize the RMSE score, that's why using -1.0.
        # In case, when you want to maximize accuracy for instance, use 1.0
        creator.create('FitnessMin', base.Fitness, weights=[-1.0])
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # create the individual operator to fill up an Individual instance:
        toolbox.register('binary', bernoulli.rvs, 0.5)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)

        # create the population operator to generate a list of individuals:
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        # genetic operators:
        toolbox.register('evaluate', self.train_evaluate)
        toolbox.register('select', tools.selTournament, tournsize=2)
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.11)
        toolbox.register('mate', tools.cxUniform, indpb=0.5)

        # create initial population (generation 0):
        population = toolbox.population(n=population_size)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        # Genetic Algorithm flow with elitism:
        population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                  ngen=max_generations, stats=stats, halloffame=hof, verbose=True)

        # print info for best solution found:
        best = hof.items[0]
        print("-- Best Individual = ", best)
        print("-- Best Fitness = ", best.fitness.values[0])

        # extract statistics:
        minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "max", "avg")

        # plot statistics:
        # sns.set_style("whitegrid")
        # plt.plot(minFitnessValues, color='blue', label="Min")
        # plt.plot(meanFitnessValues, color='green', label="Mean")
        # plt.plot(maxFitnessValues, color='red', label="Max")
        # plt.xlabel('Generation');
        # plt.ylabel('Max / Min / Average Fitness')
        # plt.legend()
        # plt.title('Max, Min and Average fitness over Generations')
        # plt.show()

        best_population = tools.selBest(population, k=k)
        return best_population
