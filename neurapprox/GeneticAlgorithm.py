from deap import base, creator, tools, algorithms
from elitism import eaSimpleWithElitism
import numpy as np
from scipy.stats import bernoulli

class GeneticAlgorithm:
    def __init__(self, hyp_dict, model, X_train, Y_train, X_val, Y_val, epochs):
        self.hyp_dict = hyp_dict
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.epochs = epochs
        self.history = []

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

    def train_evaluate(self, ga_individual_solution):
        from bitstring import BitArray
        import time
        import tensorflow as tf
        _, n_output = int(self.Y_train.shape[1])
        _, n_input = int(self.X_train.shape[1])
        t = time.time()
        t_total = 0

        # Decode GA solution to integer for window_size and num_units
        deep_layers_bits   = BitArray(ga_individual_solution[0:1])     # (8)
        num_units_bits     = BitArray(ga_individual_solution[1:2])     # (16)
        learning_rate_bits = BitArray(ga_individual_solution[2:3])    # (8)
        # #     batch_size_bits    = BitArray(ga_individual_solution[10:12])   # (4)
        # #     activation_f_bits  = BitArray(ga_individual_solution[12:13])   # (2)   Solo se consideran las 2 primeras

        deep_layers   = SC_DEEP[deep_layers_bits.uint]
        num_units     = SC_NUM_UNITS[num_units_bits.uint]
        learning_rate = SC_LEARNING[learning_rate_bits.uint]
        #     batch_size   = SC_BATCH[batch_size_bits.uint]
        #     activation_f  = SC_ACTIVATION[activation_f_bits.uint]

        # Train model and predict on validation set
        # model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(num_units, input_shape=n_input))

        for i in range(deep_layers):
           self.model.add(tf.keras.layers.Dense(num_units, activation='relu'))
        #             model.add(keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(n_output, activation=tf.nn.softmax))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, validation_data=(self.X_val, self.Y_val),
                       # callbacks=my_callbacks,
                       batch_size=128, shuffle=1, verbose=0)

        loss, score = self.model.evaluate(self.X_val, self.Y_val)
        t = time.time( ) -t
        # ss.pop(0)
        print("Accuracy:", score, ", Elapsed time:", t)
        print("-------------------------------------------------\n")

        self.history.append([deep_layers, num_units, learning_rate, loss, score, t])

        return loss,