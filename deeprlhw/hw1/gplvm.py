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
    def __init__(self, X_variational_mean, X_variational_std, Y, Kern, M , Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_variational_mean: initial latent variational distribution mean, size N (number of points) x Q (latent dimensions)
        :param X_variational_std: initial latent variational distribution std (N x Q or N x Q x Q)
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
            assert X_variational_mean.shape[1] == X_variational_std.shape[1] == X_variational_std[2]
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

    @property
    def _X_variational_conv(self):
        if tf.shape(self.X_variational_std).ndims == 3:
            return tf.matmul(self.X_variational_std, tf.transpose(self.X_variational_std, perm=[0, 2, 1]))
        elif tf.shape(self.X_variational_std).ndims == 2:
            return tf.square(self.X_variational_std)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the variational bound on 
        the marginal likelihood.
        :return: the negative of the variational bound.
        """
        num_inducing = tf.shape(self.Z)[0]
        # Compute the psi statistics.
        psi0 = tf.reduce_sum(self.kern.eKdiag(self.X_variational_mean, self._X_variational_conv), 0)
        psi1 = self.kern.eKxz(self.Z, self.X_variational_mean, self._X_variational_conv)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_variational_mean, self._X_variational_conv), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Pre-computation
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2 # Trace tricks.
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True)

        # compute the marginal log likelihood
        D = tf.cast(tf.shape(self.Y)[1], float_type)
        ND = tf.cast(tf.size(self.Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += -0.5 * D * log_det_B
        bound += tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * tf.reduce_sum(psi0) / sigma2
        bound += 0.5 * D * tf.reduce_sum(tf.matrix_diag_part(AAT))


        # compute the KL[q(x) || p(x)] TODO: the dynamics case
        KL = tf.convert_to_tensor(0, dtype=float_type)
        if tf.shape(self.X_variational_std).ndims == 2:
            # fully factorised.
            dX_variational_conv = self._X_variational_conv
            NQ = tf.cast(tf.size(self.X_variational_mean), float_type)
            KL = -0.5 * NQ + \
                 0.5 * tf.reduce_sum(tf.log(self.X_prior_var)) - \
                 0.5 * tf.reduce_sum(dX_variational_conv) + \
                 0.5 * tf.reduce_sum((tf.square(self.X_variational_mean - self.X_prior_mean) + dX_variational_conv) / self.X_prior_var)

        if tf.shape(self.X_variational_std).ndims == 3:
            # prior fully factorised, variational distribution factorised along data points
            dX_prior_var = tf.eye(self.num_latent)
            NQ = tf.cast(tf.size(self.X_variational_mean), float_type)
            L_variational = tf.cholesky(self._X_variational_conv)
            tmp = tf.cholesky(tf.matrix_triangular_solve(L_variational, tf.matrix_triangular_solve(tf.transpose(L_variational, perm=[0, 2, 1]), dX_prior_var, lower=True), lower=True))
            log_det_tmp = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(tmp)))
            KL = -0.5 * NQ \
                 + 0.5 * log_det_tmp \
                 + tf.reduce_sum(self._X_variational_conv) \
                 + 0.5 * tf.reduce_sum(tf.square(self.X_variational_mean - self.X_prior_mean))

        bound -= KL
        return bound

    def build_predict(self, Xnew, full_conv=False):
        """
        Compute the mean and variance of the latent function at some new points Xnew,
        very similar to SGPR prediction, difference is that deterministic kernel terms are replaced with
        kernel expectation w.r.t variational distribution.
        :param Xnew: Points to predict at
        """
        num_inducing = tf.shape(self.Z)[0]
        # Compute the psi statistics.
        psi1 = self.kern.eKxz(self.Z, self.X_variational_mean, self._X_variational_conv)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_variational_mean, self._X_variational_conv), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6
        Kus = self.kern.K(self.Z, Xnew)
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Pre-computation
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tf.transpose(tmp1), lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)

        if full_conv:
            var = self.kern.K(Xnew) \
                  + tf.matmul(tf.transpose(tmp1), tmp1) \
                  - tf.matmul(tf.transpose(tmp2), tmp2)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.diag(Xnew) \
                  + tf.reduce_sum(tf.square(tmp1) ,0) \
                  - tf.reduce_sum(tf.square(tmp2) ,0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            tf.tile(tf.expand_dims(var, 1), shape)

        return mean + self.mean_function(Xnew), var






