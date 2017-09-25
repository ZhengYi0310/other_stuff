'''
implementation of the ARD kernel
'''
from __future__ import division
import numpy as np
from utils import rescale,diff, sqdist, sqdist_foreach

class ARD(object):
    def __init__(self, c, b, ndim=None):
        self.log_b_ = np.log(float(b)) # signal stddev
        if c.ndim != 1:
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
        if hyper.ndim != 1:
            raise  ValueError('hyper parammeter vector for the kernel should be a vector.')
        self.log_b_ = hyper[0]
        self.log_c_ = hyper[1] if self.iso_ else hyper[1:]

    def get(self, X1, X2=None):
        # Compute the kernel between two matrix
        X1, X2 = rescale(np.exp(self.log_b_), X1, X2)
        return np.exp(self.log_b_ * 2 - 0.5 * sqdist(X1, X2))

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

    def gradX1(self, X1, X2=None):
        # Compute the derivative of the kernel w.r.t to the first argument
        # should return a m * n * d tensor
        X1, X2 = rescale(np.exp(self.log_b_), X1, X2)
        D = diff(X1, X2) # m * n * d array
        K = np.exp(self.log_b_ * 2 - 0.5 * np.sum(np.square(D), axis=-1)) # sum alsong the last axis, which is d
        G = -D * K[:, :, None] / self.log_c_ # G(m, n, d) corresponds to the derivative of of K(m ,n) w.r.t X1(m, d)
        return G

    def gradX2(self, X1, X2=None):
        # Compute the derivative of the kernel w.r.t to the second argument
        # should return a m * n * d tensor
        return -self.gradX1(X1, X2) # G(m, n, d) corresponds to the derivative of of K(m ,n) w.r.t X1(n, d)







# def createGenerator() :
#     yield 4*4
#     mylist = range(3)
#     for i in mylist:
#         yield i*i
#
# # mygenerator = createGenerator() # create a generator
# # print(mygenerator) # mygenerator is an object!
# # for i in mygenerator:
# #     print(i)
#
# a = np.array([[1, 2],[3, 4], [5, 6]])
# b = np.array([[0, 0],[1, 0]])
# c = np.array([0.1 , 0.1])
# d = np.array([[1, 2, 3]])
# # print c.ndim
# e = 0.5
# model = ARD(c ,e)
# # a = []
# # a.append([("kern.%s" % p[0],) + p[1:] for p in model._params()])
# # a.append([('mean, 1, Falsue')])
# # print a
# # hyper = np.r_[0.4, np.array([0.01, 0.01])]
# # a = np.array([[1, 2],[3, 4], [5, 6]])
# # print hyper.shape
# # model.set_hyper(hyper)
# print model.get_hyper()
#
# print len(a)
# print a.shape[0]
# print np.dot(a, c) + d[None]
# # print c
# # print np.sum(c, axis=-1)
# # print np.sum(c, axis=-1)[:,:,None].shape
# # print c * np.sum(c, axis=-1)[:,:,None]