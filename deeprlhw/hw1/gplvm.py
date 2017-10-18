import tensorflow as tf
import numpy as np
from gpflow.model import GPModel
from gpflow.gpr import GPR
from gpflow.param import Param, AutoFlow, DataHolder
from gpflow.mean_functions import Zero
from gpflow import likelihoods
from gpflow import transforms
from gpflow import kernels
from gpflow._settings import settings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist



float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

def PCA_initialization(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X to Q.
    Can be use as the initialization step for VGPDS
    :param X: data array of size N (number of points) X D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q
    """
    # assert  Q <= X.shape[1], 'Cannot have more latent dimensions than observed dimensions'
    # evals, evecs = np.linalg.eigh(np.cov(X.T))
    # i = np.argsort(evals)[::-1]
    # W = evecs[:, i]
    # W = W[:, :Q]
    # return (X - X.mean(0)).dot(W)
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
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
    def __init__(self, X_variational_mean, X_variational_var, Y, Kern, M , Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_variational_mean: initial latent variational distribution mean, size N (number of points) x Q (latent dimensions)
        :param X_variational_var: initial latent variational distribution std (N X Q)
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

        assert X_variational_var.ndim == 2, 'Incorrect number of dimensions for X_std.'
        self.X_variational_var = Param(X_variational_var, transforms.positive)
        self.num_data = X_variational_mean.shape[0]
        self.output_dim = Y.shape[1]

        assert np.all((X_variational_mean.shape == X_variational_var.shape))
        assert X_variational_mean.shape[0] == Y.shape[0], 'X variational mean and Y must be the same size.'
        assert X_variational_var.shape[0] == Y.shape[0], 'X variational std and Y must be the same size.'

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

    def build_likelihood(self):
        """
                Construct a tensorflow function to compute the bound on the marginal
                likelihood for the training data (all dimensions).
                """

        # Default: construct likelihood graph using the training data Y and initialized q(X)
        return self._build_likelihood_graph(self.X_variational_mean, self.X_variational_var, self.Y, self.X_prior_mean, self.X_prior_var)

    def _build_likelihood_graph(self, X_variational_mean, X_variational_var, Y, X_prior_mean=None, X_prior_var=None):
        """
            Construct a tensorflow function to compute the bound on the marginal
            likelihood given a Gaussian multivariate distribution representing
            X (and its priors) and observed Y
            Split from the general build_likelihood method, as the graph is reused by the held_out_data_objective
            method for inference of latent points for new data points
        """

        if X_prior_mean is None:
            X_prior_mean = tf.zeros((tf.shape(Y)[0], self.num_latent), float_type)
        if X_prior_var is None:
            X_prior_var = tf.ones((tf.shape(Y)[0], self.num_latent), float_type)
        num_inducing = tf.shape(self.Z)[0]
        # Compute the psi statistics.
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_variational_mean, X_variational_var), 0)
        psi1 = self.kern.eKxz(self.Z, X_variational_mean, X_variational_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_variational_mean, X_variational_var), 0)
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
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, Y), lower=True) / sigma

        # compute the marginal log likelihood
        D = tf.cast(tf.shape(Y)[1], float_type)
        ND = tf.cast(tf.size(Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += -0.5 * D * log_det_B
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * tf.reduce_sum(psi0) / sigma2
        bound += 0.5 * D * tf.reduce_sum(tf.matrix_diag_part(AAT))

        # compute the KL[q(x) || p(x)] TODO: the dynamics case
        # fully factorised.
        dX_variational_conv = X_variational_var if len(X_variational_var.get_shape()) == 2 else tf.matrix_diag_part(X_variational_var)
        NQ = tf.cast(tf.size(X_variational_mean), float_type)
        KL = -0.5 * NQ + \
        0.5 * tf.reduce_sum(tf.log(X_prior_var)) - \
        0.5 * tf.reduce_sum(tf.log(dX_variational_conv)) + \
        0.5 * tf.reduce_sum((tf.square(X_variational_mean - X_prior_mean) + dX_variational_conv) / X_prior_var)

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
        psi1 = self.kern.eKxz(self.Z, self.X_variational_mean, self.X_variational_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_variational_mean, self.X_variational_var), 0)
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

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (int_type, [None]))
    def held_out_data_objective(self, Ynew, mu, std, observed):
        """
        Computation of likelihood objective and gradients, given new observed points and a candidate q(X*)
        :param Ynew: new observed points, size Nnew(number of new points) x k (observed dimensions), with k <= D 
                     when k < D, it means the new points are partially observed.
        :param mu: mean for the candidate q(X*), size Nnew(num of new points) x Q (latent dimensions)
        :param var: candidate variance, np.ndarray of size Nnew (number of new points) x Q (latent dimensions) x Q (latent dimensions)
        :param observed: indices for the observed dimensions np.ndarray of size k
        :return: returning a tuple (objective,gradients). gradients is a list of 2 matrices for mu and var of size
                 Nnew x Q and Nnew x Q x Q
        """
        #TODO: partially observed points
        X_mean = tf.concat([self.X_variational_mean, mu], 0)
        X_var = tf.concat([self.X_variational_var, std], 0)
        Y = tf.concat([self.Y, Ynew], 0)

        # Build the likelihood graph for the suggested q(X,X*) and the observed dimensions of Y and Y*
        objective = self._build_likelihood_graph(X_mean, X_var, Y)

        # Collect gradients
        gradients = tf.gradients(objective, [mu, std])

        f = tf.negative(objective, name='objective')
        g = tf.negative(gradients, name='grad_objective')
        return f, g

    def _held_out_data_wrapper(self, Ynew, observed):
        """
        Private wrapper for returning an objective function which can be fit into scipy optimization module.
        :param Ynew: 
        :param observed: new observed points, size Nnew(number of new points) x k (observed dimensions), with k <= D 
                         when k < D, it means the new points are partially observed.
        :return: function accepting a flat numpy array of size 2 * Nnew (number of new points) * Q (latent dimensions)
                 and returning a tuple (objective,gradient)
        """
        half_num_params = self.latent * Ynew.shape[0] # assume fully factorised prior
        def fun(x_flat):
            # Unpack q(X*) candidate
            mu_new = x_flat[:half_num_params].reshape((Ynew.shape[0], self.num_latent))
            var_new = x_flat[half_num_params].reshape((Ynew.shape[0], self.num_latent))

            # Likelihood computation, gradients flattening
            f, g = self.held_out_data_objective(Ynew, mu_new, var_new, observed)

            return f, np.hstack(map(lambda gradients: gradients.flatten(), g))
        return fun

    def infer_latent_inputs(self, Ynew, method='L-BFGS-B', tol=None, return_logprobs=False, observed=None, **kwargs):
        """
        Compute the latent representation of the new observed inputs via maximization of the mximization of the 
        concactnated marginal likelihood.
        :param Ynew: new observed points, size Nnew(number of new points) x k (observed dimensions), with k <= D 
                     when k < D, it means the new points are partially observed.
        :param method: a string specifying the optimization rountine used by scipy.
        :param tol: the tolerance to be passed to the optimization module.
        :param return_logprobs: return the likelihood probability after optimization (default: False)
        :param observed: list specified with dimension k, if None then all dimensions D are observed.
        :param kwargs: list of options passed to the scipy optimization module.
        :returns (mean, var) or (mean, var, prob) in case return_logprobs is true.
        :rtype mean, var: np.ndarray, size Nnew (number of new points ) x Q (latent dim)
        """

        observed = np.arange(0, Ynew.shape[1]) if observed is None else np.atleast_1d(observed)
        assert(Ynew.shape[1] == observed.size)
        infer_number = Ynew.shape[0]

        # initialization of the new x based on the distance between training set Y and Ynew
        nearest_xid = np.argmin(cdist(self.Y.value[:, observed], Ynew), axis=0)
        x_init = np.hstack((self.X_variational_mean[nearest_xid, :].flatten(),
                            self.X_variational_std[nearest_xid, :].flatten()))

        f = self._held_out_data_wrapper(Ynew, observed)

        # Optimize - restrict var to be positive
        result = minimize(fun=f,
                          x0=x_init,
                          jac=True,
                          method=method,
                          tol=tol,
                          bounds=[(None, None)] * int(x_init.size / 2) + [(0, None)] * int(x_init.size / 2),
                          options=kwargs)

        x_hat = result.x
        mu = x_hat[:infer_number * self.num_latent].reshape((infer_number, self.num_latent))
        var = x_hat[infer_number * self.num_latent:].reshape((infer_number, self.num_latent))
        if return_logprobs:
            return mu, var, -result.fun
        else:
            return mu, var