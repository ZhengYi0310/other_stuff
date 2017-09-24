'''
implementation of the ARD kernel
'''
from __future__ import division
import numpy as np
from utils import rescale,diff, sqdist, sqdist_foreach

class ARD(object):
    def __init__(self, c, b, ndim=None):
        self.log_b_ = np.log(float(b)) # signal stddev
        if c.shape[0] != 1:
            raise ValueError('the length scales can only be a vector.')
        self.log_c_ = np.log(c) # lengthscales
        self.iso_ = False
        self.ndim_ = np.size(self.log_c_)
        self.nhyper_ = 1 + np.size(self.log_c_)

        if ndim is not None:
            if np.size(self.log_c_) == 1:
                self.log_c_ = float(self.log_c_)
                self.iso_ = True
                self.ndim_ = ndim

            else:
                raise ValueError('ndim only usable with scalar length scales')

    def _params(self):
        return [('signal variance', 1, True), ('length scales', self.nhyper_ - 1, True)]

    def get_hyper(self):
        return np.r_[self.log_b_, self.log_c_]

    def set_hyper(self, hyper):
        if hyper.shape[0] != 0:
            raise  ValueError('hyper parammeter vector for the kernel should be a vector.')
        self.log_b_ = hyper[0]
        self.log_c_ = hyper[1] if self.iso_ else hyper[1:]

    def get(self, X1, X2=None):
        # Compute the kernel between two matrix
        X1, X2 = rescale(np.exp(self.log_b_), X1, X2)
        return np.exp(self.log_c_ * 2 - 0.5 * sqdist(X1, X2))

    def grad(self, X1, X2=None):
        # Compute the derivative w.r.t the kernel hyper parameters
        X1, X2 = rescale(np.exp(self.log_b_), X1, X2)
        D = sqdist(X1, X2)
        K = np.exp(self.log_c_ * 2 - 0.5 * D)
        yield  2 * K # the gradient w.r.t the signal stddev
        if self.iso_:
            yield K * D # the gradient w.r.t the lengscale (square-kernel case)
        else:
            for D in sqdist_foreach(X1, X2):
                yield  K * D # the gradient w.r.t the lengscales (ARD-kernel case)




def createGenerator() :
    yield 4*4
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator() # create a generator
print(mygenerator) # mygenerator is an object!
for i in mygenerator:
    print(i)