# import deap
from deap import base, creator, tools, algorithms
from bitstring import BitArray
from elitism import eaSimpleWithElitism, main

class GeneticAlgorithm:
    def __init__(self, hyp_list):
        self.hyp_list = hyp_list

    def train_evaluate(self, ga_individual_solution):
        t = time.time()
        t_total = 0

        # list of bitArrays
        # Decode GA solution to integer for window_size and num_units
        list_bits = []
        for i in range(len(self.hyplist)):
            list_bits.append(BitArray(ga_individual_solution[i:i+1]))  # (8)

        # acá voy
        deep_layers = SC_DEEP[deep_layers_bits.uint]
        num_units = SC_NUM_UNITS[num_units_bits.uint]
        learning_rate = SC_LEARNING[learning_rate_bits.uint]
        batch_size = SC_BATCH[batch_size_bits.uint]
        #     activation_f  = SC_ACTIVATION[activation_f_bits.uint]

        # Train model and predict on validation set
        model = tf.keras.Sequential()
        #     model.add(Input(shape=(int(X_train.shape[1]),)))
        model.add(Dense(int(X_train.shape[1])))

        for i in range(deep_layers):
            model.add(Dense(num_units, activation='relu'))
        #             model.add(keras.layers.Dropout(0.3))
        model.add(Dense(2, activation='linear'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
        model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val),
                  callbacks=my_callbacks, batch_size=batch_size, shuffle=False, verbose=0)

        loss, score = model.evaluate(X_val, Y_val)
        t = time.time() - t
        ss.pop(0)
        print("Loss:", score, ", Elapsed time:", t)
        print("-------------------------------------------------\n")
        #     print(loss, score)

        datos.append([deep_layers, num_units, learning_rate, batch_size, loss, score, t])

        return loss,