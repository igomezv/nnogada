import numpy as np

class Hyperparameter:
    def __init__(self, name, values, val, vary=False):
        self.name = name
        self.values = values
        self.uint = None
        self.bitarray = None
        self.val = val
        self.vary = vary
        # self.setValues()

    def setValues(self, values):
        if type(values) == type([1, 2 ]):
            if type(values[0]) == type(1) or type(values[0]) == type(0.5):
                values = np.array(values)
        self.value = values

    # def setName(self, name):
    #     self.name = name
    #
    # def setUint(self, uint):
    #     self.uint = uint
    #
    # def setBitarray(self, bitarray):
    #     self.bitarray = bitarray
    #
    # def setDefault(self, default_val):
    #     self.default = default_val