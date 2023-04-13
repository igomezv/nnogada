import numpy as np

class Hyperparameter:
    """
    This class defines a hyperparameter object.
    """
    def __init__(self, name, values, val, vary=False):
        """
        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        values : list
            Possible values of the hyperparameter.
        val : float
            Value of the hyperparameter if vary is false.
        vary : bool
            Flag that indicates if the hyperparameter is fixed (vary=False) or not (vary=True).
        """
        self.name = name
        self.values = values
        self.uint = None
        self.bitarray = None
        self.val = val
        self.vary = vary
        self.setValues(values)

    def setValues(self, values):
        """
        Parameters
        ----------
        values : list
            list of hyperparameters
        """
        if type(values) == type([0]):
            if type(values[0]) == type(1) or type(values[0]) == type(0.5):
                values = np.array(values)
        self.values = values

    def setVal(self, new_val):
        """

        Parameters
        ----------
        new_val : float
            Set a new value for the hyperparameter object.
        """
        self.val = new_val