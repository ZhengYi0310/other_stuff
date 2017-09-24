'''
implementation of helper functions for computing distance
'''
from __future__ import division
import scipy.spatial.distance as ssd

def rescale(lengthscales, X1, X2):
    '''
    Rescale the two matrix
    '''
    X1 = X1 / lengthscales
    X2 = X2 / lengthscales if (X2 is not None) else None
    return X1, X2

def diff(X1, X2=None):
    '''
    Return the differences between the vectors in matrix 'X1' and matrix 'X2'. If 'X2' is not given
    this will return the pariwise differences in 'X1'.
    '''
    X2 = X1 if (X2 is None) else X2
    return X1[:, None, :] - X2[None, :, :]

def sqdist(X1, X2=None):
    '''
    Compute the squared Eucledean distance between 2 vectors
    '''
    X2 = X1 if (X2 is None) else X2
    return ssd.cdist(X1, X2, 'sqeuclidean')

def sqdist_foreach(X1, X2=None):
    '''
    return an iterator over each dimension returning the squared-distance
    between two sets of vector. If `X2` is not given this will iterate over the
    pairwise squared-distances in `X1` in each dimension.
    '''
    X2 = X1 if (X2 is None) else X2
    for i in xrange(X1.shape[1]):
        yield ssd.cdist(X1[:, i][: None], X2[:, i][:, None], 'sqeuclidean')



