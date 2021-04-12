import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class NeuralNet:

    def __init__(self, X, Y, topology, **kwargs):
        """
        Read the network params
        Parameters
        -----------
        X, Y: numpy array
            Data to train

        """

        self.topology = topology
        self.epochs = kwargs.pop('epochs', 50)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.batch_size = kwargs.pop('batch_size', 4)
        self.early_tol = kwargs.pop('early_tol', 100)
        psplit = kwargs.pop('psplit', 0.8)
        
    
        ntrain = int(psplit* len(X))
        indx = [ntrain]
        shuffle = np.random.permutation(len(X))
        X = X[shuffle]
        Y = Y[shuffle]
        self.X_train, self.X_test = np.split(X, indx)
        self.Y_train, self.Y_test = np.split(Y, indx) 
        
        self.model = self.model()
        self.model.summary()
        

    def model(self):
        # Red neuronal
        model = K.models.Sequential()
        # Hidden layers

        for i, nodes in enumerate(self.topology):
            if i == 0:
                model.add(K.layers.Dense(self.topology[1], input_dim=self.topology[0], activation='relu'))
            elif i < len(self.topology)-2:
                model.add(K.layers.Dense(self.topology[i+1], activation='relu'))
            else:
                model.add(K.layers.Dense(self.topology[i], activation='linear'))
        optimizer = K.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model                    
    

    def train(self):
        print("Training...")
        self.history = self.model.fit(self.X_train,
                                      self.Y_train,
                                      validation_data=(self.X_test,
                                                       self.Y_test),
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=1)
        print("Training complete!")
        return self.history
    

    def get_w_and_b(self, nlayer):
        weights, biases = self.model.layers[nlayer].get_weights()
        return weights, biases
    

    def save_model(self):
        pass


    def predict(self, x):
        if type(x) == type([1]):
            x = np.array(x)
        if type(x) == type(1):
            x = np.array([x])
        
        prediction = self.model.predict(x)
        
        return prediction
        
    def plot(self, ylogscale=False):
        plt.plot(self.history.history['loss'], label='training set')
        plt.plot(self.history.history['val_loss'], label='validation set')
        if ylogscale:
            plt.yscale('log')
        mse = np.min(self.history.history['val_loss'])
        plt.title('MSE: {} Uncertainty: {}'.format(mse, np.sqrt(mse)))
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        

