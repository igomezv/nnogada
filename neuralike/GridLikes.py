import numpy as np


class GridLikes:
    def __init__(self, like, pars_bounds, ndivs=100):
        """
        Create a grid in the parameter space and evaluate the likelihood in this grid.
        This is used to generate the training set for a neural network.

        Parameters
        ----------
        like: likelihood object
        pars: list of Parameter objects
        ndivs: number of divisions by each side of the hypercube of parameters. Default is 100
        """
        self.like = like
        self.pars_bounds = pars_bounds
        self.ndivs = ndivs
        self.len = len(pars_bounds)
        print("Generating grid of points in the parameter space...")

    # def set_slices(self, *args):
    #     return [x[(None,)*i+(slice(None),)+(None,)*(len(args)-i-1)] for i, x in enumerate(args)]

    # def makegrid(self):
    #     gridlist = []
    #     for i, bound in enumerate(self.pars_bounds):
    #         tmpdiv = np.linspace(bound[0], bound[1], self.ndivs)
    #         print(np.shape(tmpdiv))
    #         print(tmpdiv)
    #         gridlist.append(tmpdiv)
    #         print(gridlist[i], type(gridlist[i]), np.shape(gridlist[i]))
    #     return self.set_slices(*gridlist)
    def makegrid(self):
        tmp = [np.linspace(bound[0], bound[1], self.ndivs) for bound in self.pars_bounds]
        tmp_grid = np.meshgrid(*tmp)
        grid = np.array([x.flatten() for x in tmp_grid]).T
        return grid

    def like_along_axis(self, array):
        return np.apply_along_axis(self.like, 1, array)

    def make_dataset(self):
        """
        Evaluate the Likelihood function on the grid
        Returns
        -------
        Samples on the grid and their respectives likelihoods.
        """
        samples_grid = self.makegrid()
        # for s in samples_grid:
        #     print(s, type(s), np.shape(s))
        print(np.shape(samples_grid))
        print("Grid created!")
        print("Evaluating likelihoods...")
        # likes = np.array([self.like(sample) for sample in samples_grid])
        likes = self.like_along_axis(samples_grid)
        print(np.shape(samples_grid), np.shape(likes))
        return samples_grid, likes


