"""
Implementation of the Gaussian likelihood model.
"""
from __future__ import division
from __future__ import print_function

import numpy as np


class Gaussianlikelihood(object):
    """
    Likelihood model for standard Gaussian distributed errors
    """
    def __init__(self, sigma):
        self.logsigma_ = np.log(float(sigma))
        self.nhyper = 1

    def _params(self):
        return [('sigma', 1, True)]

    def _variance(self):
        """
        Simply access the noise variance
        """
        return np.exp(self.logsigma_ * 2)

    def get_hyper(self):
        return np.r_[self.logsigma_]

    def set_hyper(self, hyper):
        self.logsigma_ = hyper[0]

    def sample(self, f, rng=None):
        if isinstance(rng, int):
            np.random.seed(rng)
        return f + np.random.normal(0, self._variance(), len(f))

likelihood = Gaussianlikelihood(0.01)
print(likelihood.sample(np.array([[2,0], [0,2]]).ravel()))