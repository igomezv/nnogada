from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
import sys
from bitstring import BitArray
import time
import os
import pandas as pd
from astroNN.nn.layers import MCDropout
import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torch_optimizer import AdaBound
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from nnogada.hyperparameters import *
from tqdm import tqdm

class Nnogada:
    """
    Main class for nnogada.
    """
    def __init__(self, hyp_to_find, X_train, Y_train, X_val, Y_val,
                 regression=True, verbose=False, mcdropout=False, dropout=None, usegpu=False,
                 **kwargs):
        """
        Initialization of Nnogada class.

        Parameters
        -----------
        hyp_to_find: dict
            Dictionary with the free hyperparameters of the neural net. The names must match with the names
            in the hyperparameters.py file.
            Ex: hyperparams = {'deep': [2,3], 'num_units': [100, 200], 'batch_size': [8, 32]}

        X_train: numpy.ndarray
            Set of attributes, or independent variables, for training.

        Y_train: numpy.ndarray
            Set of labels or dependent variable for training.

        X_val: numpy.ndarray
            Set of attributes, or independent variables, for testing/validation.

        Y_test: numpy.ndarray
            Set of labels or dependent variable for testing/validation.

        regression: Boolean
            If True assumes a regression task. Else, a classification is assumed. It
            affects the default choice in the activation function for the last layer,
            if regression it is the linear function, else it is softmax.

        **kwargs: kwargs
            Optional arguments:

                deep: Hyperparameter object
                    Number of layers.
                num_units: Hyperparameter object
                    Number of nodes by layer.
                batch_size: Hyperparameter object
                    Batch size.
                learning_rate: Hyperparameter object
                    Learning rate for Adam optimizer.

                epochs: Hyperparameter object
                    Number of epochs for training.

                act_fn: Hyperparameter object
                    Activation function for the hidden layers.

                last_act_fn: Hyperparameter object
                    Activation function for the last layer.

                loss_fn: Hyperparameter object
                    Loss function.

        """
        self.neural_library = kwargs.pop('neural_library', 'keras')
        if self.neural_library == 'keras' or self.neural_library == 'tensorflow':
            if usegpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            else:
                print("Using CPU")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            # setting device on GPU if available, else CPU
            if usegpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            print('Using torch. Using device:', device)

        self.deep = kwargs.pop('deep', deep)
        self.num_units = kwargs.pop('num_units', num_units)
        self.batch_size = kwargs.pop('batch_size', batch_size)
        self.learning_rate = kwargs.pop('learning_rate', learning_rate)
        self.epochs = kwargs.pop('epochs', epochs)
        self.act_fn = kwargs.pop('act_fn', act_fn)
        self.last_act_fn = kwargs.pop('last_act_fn', last_act_fn)
        self.loss_fn = kwargs.pop('loss_fn', loss_fn)
        self.dropout = dropout
        self.mcdropout = mcdropout
        self.all_hyp_list = [self.deep, self.num_units, self.batch_size, self.learning_rate,
                             self.epochs, self.act_fn, self.last_act_fn, self.loss_fn]

        self.hyp_to_find = hyp_to_find
        self.verbose = verbose
        if regression:
            self.metric = 'mean_squared_error'
        else:
            # it is a classification problem
            self.metric = 'accuracy'
            self.last_act_fn.setVal('softmax')

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

        self.history = []
        self.best = None

    def set_hyperparameters(self):
        """
        This small routine sets as variable the hyperparameters
        indicated in the hyp_to_find dictionary.
        """
        for hyp in self.all_hyp_list:
            if hyp.name in self.hyp_to_find:
                hyp.vary = True
                hyp.setValues(self.hyp_to_find[hyp.name]) #SC_hyperparameters

    def neural_train_evaluate(self, ga_individual_solution):
        """
        This train and evaluates the neural network models with the different
        solutions proposed by the Genetic Algorithm .

        Parameters
        -----------

        ga_individual_solution:
            Individual of the genetic algorithm.

        Returns
        -------

        loss: float
            Last value for the loss function.

        """
        t = time.time()
        # Decode GA solution to integer for window_size and num_units
        hyp_vary_list = []
        self.df_colnames = []
        for i, hyp in enumerate(self.all_hyp_list):
            if hyp.vary:
                if self.verbose:
                    print(hyp.name, hyp.values, len(hyp.values))
                if len(hyp.values) <= 2:
                    nbits = 1
                elif len(hyp.values) <=4:
                    nbits = 2
                elif len(hyp.values) <=6:
                    nbits = 3
                elif len(hyp.values) <=8:
                    nbits = 4
                else:
                    sys.exit("At this moment please only use 8 possible values for hyperparameter as maximum.")
                hyp.bitarray = BitArray(ga_individual_solution[i*nbits:i*nbits+nbits])  # (8)
                hyp.setVal(hyp.values[hyp.bitarray.uint])
                hyp_vary_list.append(hyp.val)
                self.df_colnames.append(hyp.name)
                if self.verbose:
                    print(hyp.name + ": {} | ".format(hyp.val), end='')
        if self.verbose:
            print("\n-------------------------------------------------")
        if self.neural_library == 'keras':
            # Train model and predict on validation set
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(self.num_units.val, input_shape=(int(self.X_train.shape[1]),)))

            for i in range(self.deep.val):
                model.add(tf.keras.layers.Dense(self.num_units.val, activation=self.act_fn.val))
                if self.mcdropout:
                    model.add(MCDropout(0.5))
                if self.dropout:
                        model.add(tf.keras.layers.Dropout(0.3))
            # model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=tf.nn.softmax))
            model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=self.last_act_fn.val))

            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate.val, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
            # from tensorflow.keras.optimizers.legacy import Adam
            model.compile(optimizer=optimizer, loss=self.loss_fn.val, metrics=[self.metric])
            model.fit(self.X_train, self.Y_train, epochs=self.epochs.val, validation_data=(self.X_val, self.Y_val),
                      callbacks=None, batch_size=self.batch_size.val, shuffle=1, verbose=int(self.verbose))

            loss, score = model.evaluate(self.X_val, self.Y_val, verbose=int(self.verbose))
            t = time.time() - t
            if self.verbose:
                print("Loss: {:.5f} Loss: {:.5f} Elapsed time: {:.2f}".format(score, loss, t))
                print("-------------------------------------------------\n")

            # results = [hyp for hyp in hyp_vary_list].extend([loss, score, t])
            # print(results)
            self.history.append(hyp_vary_list+[loss, score, t])
            return loss,

        elif self.neural_library == 'torch':
            batch_size = int(self.batch_size.val)
            # Initialize the MLP
            self.model = MLP(int(self.X_train.shape[1]), int(self.Y_train.shape[1]), numneurons=self.num_units.val)
                             # numlayers=self.deep.val)
            self.model.apply(self.model.init_weights)
            self.model.float()
            dataset_train = LoadDataSet(self.X_train, self.Y_train)
            dataset_val = LoadDataSet(self.X_val, self.Y_val)

            trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                      num_workers=1)
            validloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                                      num_workers=1)

            # Define the loss function and optimizer
            # loss_function = nn.L1Loss()
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate.val)
            # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
            # optimizer = AdaBound(self.model.parameters(), lr=self.learning_rate, final_lr=0.01, weight_decay=1e-10, gamma=0.1)
            # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
            #                                 lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
            # it needs pytorch utilities
            if self.verbose:
                summary(self.model)
            # Run the training loop
            history_train = np.empty((1,))
            history_val = np.empty((1,))
            for epoch in range(0, self.epochs.val):
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                    # Get and prepare inputs
                    inputs, targets = data
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], targets.shape[1]))
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = self.model(inputs)
                    # Compute loss
                    loss = loss_function(outputs, targets)
                    # Perform backward pass
                    loss.backward()
                    # Perform optimization
                    optimizer.step()
                    # Print statistics
                    current_loss += loss.item()
                    if i % 10 == 0:
                        current_loss = 0.0
                history_train = np.append(history_train, current_loss)

                valid_loss = 0.0
                self.model.eval()  # Optional when not using Model Specific layer
                for i, data in enumerate(validloader, 0):
                    # Get and prepare inputs
                    inputs, targets = data
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], targets.shape[1]))
                    output_val = self.model(inputs)
                    valid_loss = loss_function(output_val, targets)
                    valid_loss += loss.item()

                history_val = np.append(history_val, valid_loss.item())
                if self.verbose:
                    print('Epoch: {}/{} | Training Loss: {:.5f} | Validation Loss:'
                          '{:.5f}'.format(epoch + 1, self.epochs.val, loss.item(), valid_loss.item()), end='\r')

            t = time.time() - t
            if self.verbose:
                print('\nTraining process has finished in {:.2f} minutes.'.format(t / 60))
                print("-------------------------------------------------\n")
            # history = {'loss': history_train, 'val_loss': history_val}
            self.loss_val = history_val[-5:]
            self.loss_train = history_train[-5:]
            # print("current loss: {} valid_loss: {} loss_val: {}".format(loss, valid_loss, self.loss_val[-1]))
            self.history.append(hyp_vary_list + [float(loss), float(valid_loss), t])
            return self.loss_val[-1],

    def eaSimpleWithElitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                            halloffame=None, pbar=None):
        """
        Method based on https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python.

        The individuals contained in the halloffame are directly injected into the next generation and are not subject to the
        genetic operators of selection, crossover and mutation.

        Parameters
        ----------
        population : list
            List of individuals.
        toolbox : deap.base.Toolbox object
            Toolbox that contains the genetic operators.
        cxpb : float
            The probability of crossover between two individuals.
        mutpb : float
            Probability of mutation.
        ngen : int
            Number of generation.
        stats : deap.tools.Statistics object
             A Statistics object that is updated inplace, optional.
        halloffame : deap.tools.HallOfFame object
            Object that will contain the best individuals, optional.
        pbar : bool
            Flag to use progres bar with tqdm library.

        Returns
        -------
        population : list
            List of individuals.
        logbook : deap.tools.Logbook object.
         Statistics of the evolution.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # nnogada: it evaluates all the individuals.
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            if pbar:
                pbar.update(1)

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
            if self.verbose:
                print(logbook.stream)

        return population, logbook

    def ga_with_elitism(self, population_size, max_generations, gene_length, k,
                        pmutation=0.5, pcrossover=0.5, hof=1):
        """
            Simple genetic algorithm with elitism.

            Parameters
            -----------
            population_size : int
                Population size.
            max_generations : int
                Maximum number of generations.
            gene_length : int
                Length of each gene.
            k : int
                k parameter for the tournament selection method
            pmutation : float
                Probability of mutation, between 0 and 1.
            pcrossover : float
                Probability of crossover, between 0 and 1.
            hof : int
                Number of individuals to stay in the hall of fame.

            Returns
            -------
            best_population : list
                Individuals in the last population.
        """
        # Genetic Algorithm constants:
        P_CROSSOVER = pcrossover  # probability for crossover
        P_MUTATION = pmutation  # probability for mutating an individual
        HALL_OF_FAME_SIZE = hof  # Best individuals that pass to the other generation
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
        toolbox.register('evaluate', self.neural_train_evaluate)
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
        try:
            pbar = tqdm(total=max_generations)
        except:
            pbar = None
        population, logbook = self.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                       ngen=max_generations, stats=stats, halloffame=hof, pbar=pbar)

        best_population = tools.selBest(population, k=k)
        # convert the history list in a data frame
        self.df_colnames = self.df_colnames + ['loss', 'score', 'time']
        self.history = pd.DataFrame(self.history, columns=self.df_colnames)
        self.history = self.history.sort_values(by='loss', ascending=True, ignore_index=True)
        print("\nBest 5 solutions:\n-----------------\n")
        print(self.history.head(5))
        self.best = self.history.iloc[0]

        return best_population

# for torch nets
class LoadDataSet:
    def __init__(self, X, y, scale_data=False):
        """
        Prepare the dataset for regression

        Parameters
        ----------
        X : numpy.darray
            Input data for the neural network training.
        y : numpy.darray
            Output data for the neural network training.
        scale_data : bool
            Flag to scale the training data.
        """
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # # Apply scaling if necessary
            # if scale_data:
            #     X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    """
    Multilayer Perceptron class for regression.
    """
    def __init__(self, ncols, noutput, numneurons=200,
                 numlayers=3, dropout=0.5):
        """
        Initialization method.

        Parameters
        ----------
        ncols : int
            Number of attributes.
        noutput : int
            Size of the output.
        numneurons : int
            Number of neurons for the hidden layers.
        numlayers : int
            Number of hidden layers.
        dropout : float
            Dropout value.
        """
        super().__init__()

        l_input = nn.Linear(ncols, numneurons)
        a_input = nn.ReLU()

        l_hidden = nn.Linear(numneurons, numneurons)
        a_hidden = nn.ReLU()

        l_output = nn.Linear(numneurons, noutput)

        l = [l_input, a_input]
        for _ in range(numlayers):
            l.append(l_hidden)
            l.append(a_hidden)
        l.append(l_output)
        self.module_list = nn.ModuleList(l)

    def forward(self, x):
        """
        Forward method using activation function and other functions defined in the torch architecture.

        Parameters
        ----------
        x : numpy.array
            Input array.

        Returns
        -------
        x : numpy.array
            Array before a forward step.

        """
        for f in self.module_list:
            x = f(x)
        return x

    def init_weights(self, m):
        """
        Initilization of the ANN weights.

        Parameters
        ----------
        m : MLP class.
            Multilayer perceptron model.
        """
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
