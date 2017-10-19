import numpy as np
import tensorflow as tf
from gpflow.gplvm import BayesianGPLVM
from gpflow.model import Model
from gpflow.param import DataHolder, ParamList, Param, AutoFlow
from gpflow import likelihoods, transforms
from gpflow._settings import settings

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64
class BayesianDGPLVM(Model):
    def __init__(self, X_variational_mean, X_variational_var, Y, kern, t, kern_t, M , Z=None):
        """
        Initialization of Bayesian Gaussian Process Dynamics Model. This method only works with Gaussian likelihood.
        :param X_variational_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_variational_var: variance of latent positions (N x Q), for the initialisation of the latent space.
        :param Y: data matrix, size N (number of points) x D (dimensions).
        :param kern: kernel specification, by default RBF.
        :param t: time stamps.
        :param kern_t: dynamics kernel specification, by default RBF.
        :param M: number of inducing points.
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions), By default
                  random permutation of X_mean.
        """
        super(BayesianDGPLVM, self).__init__(name='BayesianDGPLVM')
        self.kern = kern
        assert np.all((X_variational_mean.shape == X_variational_var.shape))
        assert X_variational_mean.shape[0] == Y.shape[0], 'X variational mean and Y must be the same size.'
        assert X_variational_var.shape[0] == Y.shape[0], 'X variational std and Y must be the same size.'
        self.likelihood = likelihoods.Gaussian()

        # multiple sequences
        series = []
        for i in range(len(X_variational_mean)):
            series.append(GPTimeSeries(X_variational_mean[i], X_variational_var[i], t[i]))
        self.series = ParamList(series)

        # inducing points
        if Z is None:
            # By default we initialize by permutation of initial
            Z = np.random.permutation(np.concatenate(X_variational_mean, axis=0).copy())[:M]
        else:
            assert Z.shape[0] == M
        self.Z = Param(Z)

        self.kern_t = kern_t
        self.Y = DataHolder(Y)
        self.M = M
        self.n_s = 0

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        num_inducing = self.M
        psi0 = .0
        psi1 = None
        psi2 = .0
        KL = .0

        for s in self.series:
            KLs, psi0s, psi1s, psi2s = s.build_likelihood(self.Z, self.kern, self.kern_t)
            KL += KLs
            psi0 += psi0s
            if psi1 is None:
                psi1 = psi1s
            else:
                psi1 = tf.concat([psi1, psi1s], 0)

        # build the log likelihood
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # pre-computation
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1)) + tf.eye(num_inducing) * 1e-6
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2 # Trace tricks
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

        bound -= KL

        return bound


class GPTimeSeries(Model):
    def __init__(self, X_variational_mean, X_varialtional_var, t):
        """
        
        :param X_variational_mean: initial latent variational distribution mean, size N (number of points) x Q (latent dimensions)
        :param X_variational_var: initial latent variational distribution std (N x Q)
        :param t: time stamps for the variational prior kernel, need to ba an np.narray. 
        """
        super(GPTimeSeries, self).__init__(name='GPTimeSeries')
        self.X_variational_mean = Param(X_variational_mean)
        self.X_variational_var = Param(X_varialtional_var)
        assert self.X_variational_var == 2, "the dimensionality of variational prior covariance needs to be 2."
        assert np.all(self.X_variational_mean.shape == self.X_variational_var.shape), "the shape of variational prior mean and variational prior covariance needs to be equal."
        self.num_latent = X_variational_mean.shape[1]
        self.num_data = X_variational_mean.shape[0]
        assert (isinstance(t, np.ndarray)), "time stamps need to be a numpy array."
        t = DataHolder(t)
        self.t = t

    def build_likelihood(self, Z, kern, kern_t, give_KL=True):
        """
        
        :param Z: inducing points 
        :param kern: kernel for the q(X)
        :param kern_t: kernel for the p(X)
        :param give_KL: 
        :return: 
        """
        # "The Dynamical Variational GP-LVM for Sequence Data" part in sec 3.3 of Andreas Damianou's Phd thesis.
        #########################################
        Kxx = kern_t.K(self.t) + tf.eye(self.num_data, dtype=float_type) * 1e-6 # N x N, prior covariance for p(X)
        Lx = tf.cholesky(Kxx)
        Lambda = tf.matrix_diag(tf.transpose(self.X_variational_var)) # Q x N x N, prior covariance for q(X)
        tmp = tf.eye(self.num_data) + tf.einsum('ijk,kl->ijl', tf.einsum('ij,kil->kjl', Lx, Lambda), Lx) # I + Lx^T x Lambda x Lx in batch mode
        Ltmp = tf.cholesky(tmp) # Q x N x N
        tmp2 = tf.matrix_triangular_solve(Ltmp, tf.tile(tf.expand_dims(tf.transpose(Lx), 0), tf.stack([self.num_latent, 1, 1])))
        S_full = tf.einsum(('ijk,ijl->ikl', tmp2, tmp2)) # Q x N x N
        S = tf.transpose(tf.matrix_diag_part(S_full)) # N x Q, marginal distribution of multivariate normal, from column-wise to row-wise.
        mu = tf.matmul(Kxx, self.X_variational_mean) # N x Q
        ##########################################

        psi0 = tf.reduce_sum(kern.eKdiag(mu, S), 0) # N
        psi1 = self.kern.eKxz(Z, mu, S) # N x M
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, mu, S), 0) # N x M x M

        # compute the KL[q(X) || p(X)]
        NQ = tf.cast(tf.size(mu), float_type)
        if give_KL:
            KL = -0.5 * NQ
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(Ltmp))) # trace tricks
            KL += 0.5 * tf.reduce_sum(tf.trace(tf.cholesky_solve(tf.tile(tf.expand_dims(Lx, 0),
                                               tf.stack([self.num_latent, 1, 1])) , S_full + tf.einsum('ji,ki->ijk', mu, mu))))
            return KL, psi0, psi1, psi2
        else:
            return psi0, psi1, psi2

    def build_latent(self, kern_t, full_conv=False):
        Kxx = kern_t.K(self.t) + tf.eye(self.num_data, dtype=float_type) * 1e-6  # N x N, prior covariance for p(X)
        Lx = tf.cholesky(Kxx)
        Lambda = tf.matrix_diag(tf.transpose(self.X_variational_var))  # Q x N x N, prior covariance for q(X)
        tmp = tf.eye(self.num_data) + tf.einsum('ijk,kl->ijl', tf.einsum('ij,kil->kjl', Lx, Lambda),
                                                Lx)  # I + Lx^T x Lambda x Lx in batch mode
        Ltmp = tf.cholesky(tmp)  # Q x N x N
        tmp2 = tf.matrix_triangular_solve(Ltmp, tf.tile(tf.expand_dims(tf.transpose(Lx), 0),
                                                        tf.stack([self.num_latent, 1, 1])))
        if full_conv:
            S = tf.einsum(('ijk,ijl->ikl', tmp2, tmp2))  # Q x N x N
        else:
            S = tf.transpose(tf.matrix_diag_part(tf.einsum(('ijk,ijl->ikl', tmp2, tmp2))))  # N x Q, marginal distribution of multivariate normal, from column-wise to row-wise.
        mu = tf.matmul(Kxx, self.X_variational_mean)  # N x Q
        return mu, S

    def build_predict(self, t_new, kern_t):
        Kno = kern_t.K(t_new, self.t)
        Koo = kern_t.K(self.t)
        Knn = kern_t.K(t_new)

        mu_xn = tf.matmul(Kno, self.X_variational_mean) # Nnew x Q
        tmp1 = Koo + tf.matrix_diag(tf.divide(1, self.X_variational_var)) # Q x N x N
        Ltmp1 = tf.cholesky(tmp1)
        tmp2 = tf.matrix_triangular_solve(Ltmp1, tf.tile(tf.expand_dims(Kno, 0), tf.stack([self.num_latent, 1, 1]))) # Q x N x N
        var_xn = Knn - tf.einsum(('ijk,ijl->ikl', tmp2, tmp2)) # Q x Nnew x Nnew
        var_xn = tf.transpose(tf.matrix_diag_part(var_xn))

        return mu_xn, var_xn


