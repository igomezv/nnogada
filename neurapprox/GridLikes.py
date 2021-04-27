import os
import numpy as np
import time


class GridLikes:
    def __init__(self, function, pars_bounds, ndivs=5, pool=None, files_path='grid'):
        """
        Create a grid in the parameter space and evaluate the function in this grid.
        This is used to generate the training set for a neural network.

        Parameters
        ----------
        function: target function
        pars: list of Parameter objects
        ndivs: number of divisions by each side of the hypercube of parameters. Default is 100
        """
        self.function = function
        self.pars_bounds = pars_bounds
        self.ndivs = ndivs
        self.len = len(pars_bounds)
        self.pool = pool
        self.files_path = files_path
        if pool:
            self.M = pool.map
        else:
            self.M = map
        print("Generating grid of points in the parameter space...")

    def makegrid(self):
        if not self.filesChecker():
            tmp = [np.linspace(bound[0], bound[1], self.ndivs) for bound in self.pars_bounds]
            tmp_grid = np.meshgrid(*tmp)
            grid = np.array([x.flatten() for x in tmp_grid]).T
            np.save('{}_grid.npy'.format(self.files_path), grid)
        else:
            print('Loading existing grid and functions: {}'.format(self.files_path))
            grid = np.load('{}_grid.npy'.format(self.files_path))
        print("Grid of points in the parameter space created!")
        return grid

    def function_along_axis(self, array):
        # under test
        return np.apply_along_axis(self.function, 1, array)

    def make_dataset(self):
        """
        Evaluate the function function on the grid
        Returns
        -------
        Samples on the grid and their respectives functions.
        """
        samples_grid = self.makegrid()
        t1 = time.time()
        if not self.filesChecker():
            print("Evaluating functions...")
            functions = np.array(list(self.M(self.function, samples_grid)))
            np.save('{}_functions.npy'.format(self.files_path), functions)
        else:
            print('Loading existing grid and functions: {}'.format(self.files_path))
            functions = np.load('{}_functions.npy'.format(self.files_path))
        # functions = np.array([self.function(x) for x in samples_grid])
        # functions = self.function_along_axis(samples_grid)
        tf = time.time() - t1
        print("Time of {} function evaluations {:.4f} min".format(len(functions), tf/60))
        print("Training dataset created!")
        if self.pool:
            self.pool.close()
        # print("Time of evaluating {} functions with apply_along_axis: {:.4} s".format(len(functions), tf))

        return samples_grid, functions

    def filesChecker(self):
        """
        This method checks if the name of the grid.npy and functions.npy exists, if it already exists use it
        """
        if os.path.isfile('{}_grid.npy'.format(self.files_path)):
            if os.path.isfile('{}_functions.npy'.format(self.files_path)):
                return True
        else:
            return False

