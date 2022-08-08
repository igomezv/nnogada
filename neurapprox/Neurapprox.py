from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
import time
import tensorflow as tf
from neurapprox.hyperparameters import *

class Neurapprox:
    def __init__(self, hyp_to_find, X_train, Y_train, X_val, Y_val):
        self.deep = deep
        self.num_units = num_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.hyp_to_find = hyp_to_find
        # hyp_dict keys: name, values (np.array([3,4])),
        # self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.history = []

    def setHyperparameters(self):
        """
        dict_hyp:
        {'num_units': [1,2,3], ''}
        """
        for hyp in all_hyp_list:
            if hyp.name in self.hyp_to_find:
                hyp.vary = True
                hyp.setValues(self.hyp_to_find[hyp.name]) #SC_hyperparameters


    # def train_evaluate(self, ga_individual_solution):
    #     n_output = int(self.Y_train.shape[1])
    #     n_input = int(self.X_train.shape[1])
    #     print(n_input)
    #     t = time.time()
    #     t_total = 0
    #     i = 0
    #     for hyp in all_hyp_list:
    #         if hyp.vary is True:
    #             print(i, hyp.name)
    #             hyp.bitarray = BitArray(ga_individual_solution[i:i+1])  # (8)
    #             # hyp.val = hyp.values[hyp.bitarray.uint]
    #             hyp.setVal(hyp.values[hyp.bitarray.uint])
    #             print(type(hyp.values), np.shape(hyp.values))
    #             i += 1
    #
    #
    #     # Train model and predict on validation set
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(num_units.val, input_shape=n_input))
    #
    #     for i in range(deep.val):
    #         model.add(tf.keras.layers.Dense(num_units.val, activation='relu'))
    #     #             model.add(keras.layers.Dropout(0.3))
    #     model.add(tf.keras.layers.Dense(n_output, activation=tf.nn.softmax))
    #
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate.val, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
    #     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    #     model.fit(self.X_train, self.Y_train, epochs=epochs.val, validation_data=(self.X_val, self.Y_val),
    #                    # callbacks=my_callbacks,
    #                    batch_size=batch_size.val, shuffle=1, verbose=0)
    #
    #     loss, score = model.evaluate(self.X_val, self.Y_val)
    #     t = time.time( ) -t
    #     # ss.pop(0)
    #     print("Accuracy:", score, ", Elapsed time:", t)
    #     print("-------------------------------------------------\n")
    #
    #     self.history.append([deep.val, num_units.val, batch_size.val, learning_rate.val, loss, score, t])
    #
    #     return loss, model

    def train_evaluate(self, ga_individual_solution):
        t = time.time()
        t_total = 0
        datos = []

        # Decode GA solution to integer for window_size and num_units
        deep_layers_bits = BitArray(ga_individual_solution[0:1])  # (8)
        num_units_bits = BitArray(ga_individual_solution[1:2])  # (16)
        learning_rate_bits = BitArray(ga_individual_solution[2:3])  # (8)
        # #     batch_size_bits    = BitArray(ga_individual_solution[10:12])   # (4)
        # #     activation_f_bits  = BitArray(ga_individual_solution[12:13])   # (2)   Solo se consideran las 2 primeras

        deep_layers_uint = deep.values[deep_layers_bits.uint]
        # print(deep_layers, np.shape(deep_layers))
        num_units_uint = num_units.values[num_units_bits.uint]
        learning_rate_uint = learning_rate.values[learning_rate_bits.uint]
        #     batch_size   = SC_BATCH[batch_size_bits.uint]
        #     activation_f  = SC_ACTIVATION[activation_f_bits.uint]

        #     print('\n--------------- Starting trial:', population_size*(max_generations+1)-len(ss), "---------------")
        print('Deep layers:', deep_layers_uint, ', Number of neurons:', num_units_uint, ", Learning rate:", learning_rate_uint)
        #     print("-------------------------------------------------")

        # Train model and predict on validation set
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Input(shape=(int(self.X_train.shape[1]),)))
        model.add(tf.keras.layers.Dense(num_units_uint, input_shape=(int(self.X_train.shape[1]),)))

        for i in range(deep_layers_uint):
            model.add(tf.keras.layers.Dense(num_units_uint, activation='relu'))
        #             model.add(keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_uint, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
        model.fit(self.X_train, self.Y_train, epochs=epochs.val, validation_data=(self.X_val, self.Y_val),
                  callbacks=None, batch_size=128, shuffle=1, verbose=0)

        loss, score = model.evaluate(self.X_val, self.Y_val)
        t = time.time() - t
        #     ss.pop(0)
        print("Accuracy:", score, ", Elapsed time:", t)
        print("-------------------------------------------------\n")
        #     print(loss, score)

        datos.append([deep_layers_uint, num_units_uint, learning_rate_uint, loss, score, t])

        return loss,
    def eaSimpleWithElitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
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
        population, logbook = self.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                  ngen=max_generations, stats=stats, halloffame=hof, verbose=True)

        # print info for best solution found:
        best = hof.items[0]
        print("-- Best Individual = ", best)
        print("-- Best Fitness = ", best.fitness.values[0])

        # extract statistics:
        minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "max", "avg")

        # # plot statistics:
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
        return best_population