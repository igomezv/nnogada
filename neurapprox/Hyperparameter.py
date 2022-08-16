import numpy as np

class Hyperparameter:
    def __init__(self, name, values, val, vary=False):
        self.name = name
        self.values = values
        self.uint = None
        self.bitarray = None
        self.val = val
        self.vary = vary
        self.setValues(values)

    def setValues(self, values):
        if type(values) == type([0]):
            if type(values[0]) == type(1) or type(values[0]) == type(0.5):
                values = np.array(values)
        self.values = values

    def setVal(self, new_val):
        self.val = new_val