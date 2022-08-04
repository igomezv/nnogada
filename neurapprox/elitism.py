from deap import algorithms
from deap import tools

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)
        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)
        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# # Genetic Algorithm flow:
# def main():
#
#     # create initial population (generation 0):
#     population = toolbox.population(n=population_size)
#
#     # prepare the statistics object:
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("min", np.min)
#     stats.register("avg", np.mean)
#
#     # define the hall-of-fame object:
#     hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
#
#     # perform the Genetic Algorithm flow with elitism:
#     population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
#                                               ngen=max_generations, stats=stats, halloffame=hof, verbose=True)
#
#     # print info for best solution found:
#     best = hof.items[0]
#     print("-- Best Individual = ", best)
#     print("-- Best Fitness = ", best.fitness.values[0])
#
#     # extract statistics:
#     minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
#
#     # plot statistics:
#     sns.set_style("whitegrid")
#     plt.plot(minFitnessValues, color='red')
#     plt.plot(meanFitnessValues, color='green')
#     plt.xlabel('Generation')
#     plt.ylabel('Min / Average Fitness')
#     plt.title('Min and Average fitness over Generations (log scale)')
#     plt.yscale("log")
#     plt.show()

# print("Elitism succesfully imported")