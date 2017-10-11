import tensorflow as tf
import numpy as np
from gpflow.model import GPModel
from gpflow.gpr import GPR
from gpflow.param import Param
from gpflow import  kullback_leiblers
from gpflow.mean_functions import Zero
from gpflow import likelihoods
from gpflow import transforms
from gpflow import kernels
from gpflow._settings import settings

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

def PCA_initialization(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X to Q.
    Can be use as the initialization step for VGPDS
    :param X: data array of size N (number of points) X D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q
    """
    assert  Q <= X.shape[1], 'Cannot have more latent dimensions than observed dimensions'
    evals, evecs = np.linalg.eigh(np.conv(X.T))
    i = np.argsort(evals)[::-1]
    W = evecs[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)

class GPLVM(GPR):
    """
        Standard GPLVM where the likelihood can be optimised with respect to the latent X.
        """

    def __init__(self, Y, latent_dim, X_mean=None, kern=None, mean_function=Zero()):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions)
        :param X_mean: latent positions (N x Q), for the initialisation of the latent space.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """
        if kern is None:
            kern = kernels.RBF(latent_dim, ARD=True)
        if X_mean is None:
            X_mean = PCA_initialization(Y, latent_dim)
        assert X_mean.shape[1] == latent_dim, \
            'Passed in number of latent ' + str(latent_dim) + ' does not match initial X ' + str(X_mean.shape[1])
        self.num_latent = X_mean.shape[1]
        assert Y.shape[1] >= self.num_latent, 'More latent dimensions than observed.'
        GPR.__init__(self, X_mean, Y, kern, mean_function=mean_function)
        del self.X  # in GPLVM this is a Param
        self.X = Param(X_mean)


class BayesianGPLVM(GPModel):
    def __init__(self, X_variational_mean, X_variational_std, Y, Kern, M , Z=None, X_prior_mean=None, X_prior_var = None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_variational_mean: initial latent variational distribution mean, size N (number of points) x Q (latent dimensions)
        :param X_variational_std: initial latent variational distribution std (N x Q or N x N x Q)
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param Kern: kernal specification, by default RBF-ARD
        :param M: number of inducing points 
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
                  random permutation of X_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        GPModel.__init__(self, X_variational_mean, Y, Kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X # in GPLVM this is a Param
        self.X_variational_mean = Param(X_variational_mean)

        if X_variational_std.ndim == 2:
            self.X_variational_std = Param(X_variational_std, transforms.positive)
        elif X_variational_std.ndim == 3: # full covariance matrix.
            self.X_variational_std = Param(X_variational_std)
        else:
            raise AssertionError("Incorrect number of dimensions for X_std.")

        self.num_data = X_variational_mean.shape[0]
        self.output_dim = Y.shape[1]

        if X_variational_std.ndim == 2:
            assert (X_variational_mean.shape == X_variational_std.shape)
        elif X_variational_std.ndim == 3:
            assert X_variational_mean.shape[0] == X_variational_std.shape[0] == X_variational_std[1]
        assert X_variational_mean.shape[0] == Y.shape[0], 'X variational mean and Y must be the same size.'
        assert X_variational_std.shape[0] == Y.shape[0], 'X variational std and Y must be the same size.'

        # inducing points
        if Z is None:
            # By default it's initialized by random permutation of the latent inputs.
            Z = np.random.permutation(X_variational_mean.copy())[:M]
        else:
            assert Z.shape[0] == M, 'Only M inducing points are allowed, however {} are provided.'.format(Z.shape[0])
        self.Z = Param(Z)
        self.num_latent = Z.shape[1]
        assert X_variational_mean.shape[1] == self.num_latent

        # Prior mean and variance for X TODO: the dynamic case is different
        if X_prior_mean is None:
            X_prior_mean = np.zeros((self.num_data, self.num_latent))
        self.X_prior_mean = X_prior_mean
        if X_prior_var is None:
            X_prior_var = np.ones((self.num_data, self.num_latent))
        self.X_prior_var = X_prior_var

        assert X_prior_var.shape[0] == self.num_data
        assert X_prior_var.shape[1] == self.num_latent
        assert X_prior_mean.shape[0] == self.num_data
        assert X_prior_mean.shape[1] == self.num_latent