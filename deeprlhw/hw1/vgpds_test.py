import numpy as np
import tensorflow as tf
from gpflow.gplvm import BayesianGPLVM
from gpflow.model import Model
from gpflow.param import DataHolder, ParamList, Param, AutoFlow
from gpflow import transforms, likelihoods
from gpflow._settings import settings
float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class BayesianDGPLVM(Model):
    def __init__(self, time, kern_t, X_mean, X_var, Y, kern, M, Z=None):
        """
        Initialise Bayesian DGPLVM object. This method only works with a Gaussian likelihood.
        :param X_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_var: variance of latent positions (N x Q), for the initialisation of the latent space.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param kern: kernel specification, by default RBF
        :param M: number of inducing points
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
        random permutation of X_mean.
        :param kern_t: dynamical kernel
        """
        super(BayesianDGPLVM, self).__init__('DGPLVM')
        self.kern = kern
        assert len(X_mean) == len(X_var), 'must be same amount of time series'
        self.likelihood = likelihoods.Gaussian()
        series = []
        for i in range(len(X_mean)):
            series.append(TimeSeries(X_mean[i], X_var[i], time[i]))

        self.series = ParamList(series)
        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            Z = np.random.permutation(np.concatenate(X_mean, axis=0).copy())[:M]
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
            psi2 += psi2s
            if psi1 is None:
                psi1 = psi1s
            else:
                psi1 = tf.concat([psi1, psi1s], 0)


        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-5
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # compute log marginal bound
        D = tf.cast(tf.shape(self.Y)[1], float_type)
        ND = tf.cast(tf.size(self.Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.diag_part(AAT)))
        bound -= KL

        return bound

    def build_latent(self, full_cov=False):
        S = []
        m = []
        for s in self.series:
            ms, Ss = s.build_latent(self.kern_t, full_cov)
            m.append(ms)
            S.append(Ss)
        return m, S

    def build_predict(self, t_new, full_cov=False):
        n_s = self.n_s
        psi0 = .0
        psi1 = None
        psi2 = .0

        for s in self.series:
            psi0s, psi1s, psi2s = s.build_likelihood(self.Z, self.kern, self.kern_t, give_KL=False)
            psi0 += psi0s
            psi2 += psi2s
            if psi1 is None:
                psi1 = psi1s
            else:
                psi1 = psi1 = tf.concat([psi1, psi1s], 0)

        mu_xn, var_xn = self.series[n_s].build_predict(t_new, self.kern_t)

        num_inducing = tf.shape(self.Z)[0]
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        psi0 = self.kern.eKdiag(mu_xn, var_xn)
        psi1 = tf.transpose(self.kern.eKxz(self.Z, mu_xn, var_xn))

        tmp1 = tf.matrix_triangular_solve(L, psi1, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)

        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = psi0 + tf.matmul(tf.transpose(tmp2), tmp2) \
                  - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = psi0 + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)

        return mean, var

    def predict_serie(self, t_new, noise=False, full=False, n_s=0):
        self.n_s = n_s
        if noise:
            return self.predict_y(t_new)
        elif full:
            return self.predict_f_full_cov(t_new)
        else:
            return self.predict_f(t_new)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, t_new):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(t_new, False)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, t_new):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        pred = self.build_predict(t_new, False)
        return pred

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, t_new):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        pred = self.build_predict(t_new, True)
        return pred

    @AutoFlow()
    def get_latent(self, full_cov=False):
        return self.build_latent(full_cov)

    @AutoFlow((float_type, [None, None]))
    def predict_from_latent(self, Xnew, full_cov=False):
        psi0 = .0
        psi1 = None
        psi2 = .0
        KL = .0

        for s in self.series:
            KLs, psi0s, psi1s, psi2s = s.build_likelihood(self.Z, self.kern, self.kern_t)
            KL += KLs
            psi0 += psi0s
            psi2 += psi2s
            if psi1 is None:
                psi1 = psi1s
            else:
                psi1 = psi1 = tf.concat([psi1, psi1s], 0)

        Kus = self.kern.K(self.Z, Xnew)

        num_inducing = tf.shape(self.Z)[0]
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2) \
                  - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var


class TimeSeries(Model):
    def __init__(self, X_mean, X_var, t):
        super(TimeSeries, self).__init__()
        self.X_mean = Param(X_mean)
        assert X_var.ndim == 2
        self.X_var = Param(X_var, transforms.positive)
        assert np.all(X_mean.shape == X_var.shape)
        self.num_latent = X_mean.shape[1]
        self.num_data = t.shape[0]
        if isinstance(t, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            t = DataHolder(t)
        self.t = t

    def build_likelihood(self, Z, kern, kern_t, give_KL=True):
        Kxx = kern_t.K(self.t) + tf.eye(self.num_data, dtype=float_type) * 1E-5  # n x n
        Lx = tf.cholesky(Kxx)
        Lambda = tf.matrix_diag(tf.transpose(self.X_var))  # q x n x n
        tmp = tf.eye(self.num_data, dtype=float_type) + tf.einsum('ijk,kl->ijl', tf.einsum('ij,kil->kjl', Lx, Lambda), Lx)  # q x n x n
        Lt = tf.cholesky(tmp)  # q x n x n
        tmp2 = tf.matrix_triangular_solve(Lt,
                                          tf.tile(tf.expand_dims(tf.transpose(Lx), 0),
                                                  tf.stack([self.num_latent, 1, 1])))  # q x n x n
        S_full = tf.einsum('ikj,ikl->ijl', tmp2, tmp2)
        S = tf.transpose(tf.matrix_diag_part(S_full))  # n x q
        mu = tf.matmul(Kxx, self.X_mean)  # n x q

        psi0 = tf.reduce_sum(kern.eKdiag(mu, S), 0)  # N
        psi1 = kern.eKxz(Z, mu, S)  # N x M
        psi2 = tf.reduce_sum(kern.eKzxKxz(Z, mu, S), 0)  # N x M x M

        # KL[q(x) || p(x)]
        NQ = tf.cast(tf.size(mu), float_type)

        muT = tf.transpose(mu)  # q x ns
        if give_KL:
            KL = 0.5 * tf.reduce_sum(
                tf.trace(tf.cholesky_solve(tf.tile(tf.expand_dims(Lx, 0), [self.num_latent, 1, 1]),
                                           S_full + tf.einsum('ij,ik->ijk', muT, muT)))) + tf.reduce_sum(
                tf.log(tf.matrix_diag_part(Lt))) - NQ * 0.5
            return KL, psi0, psi1, psi2
        else:
            return psi0, psi1, psi2

    def build_latent(self, kern_t, full_cov=False):
        Kxx = kern_t.K(self.t) + tf.eye(self.num_data, dtype=float_type) * 1E-5  # n x n
        Lx = tf.cholesky(Kxx)
        Lambda = tf.matrix_diag(tf.transpose(self.X_var))  # q x n x n
        tmp = tf.eye(self.num_data) + tf.einsum('ijk,kl->ijl', tf.einsum('ij,kil->kjl', Lx, Lambda), Lx)  # q x n x n
        Lt = tf.cholesky(tmp)  # q x n x n
        tmp2 = tf.matrix_triangular_solve(Lt,
                                          tf.tile(tf.expand_dims(tf.transpose(Lx), 0),
                                                  tf.stack([self.num_latent, 1, 1])))  # q x n x n
        if full_cov:
            S = tf.einsum('ikj,ikl->ijl', tmp2, tmp2)
        else:
            S = tf.transpose(tf.matrix_diag_part(tf.einsum('ikj,ikl->ijl', tmp2, tmp2)))
        m = tf.matmul(Kxx, self.X_mean)  # n x q
        return m, S

    def build_predict(self, t_new, kern_t):
        Kno = kern_t.K(t_new, self.t)
        Koo = kern_t.K(self.t)
        Knn = kern_t.K(t_new)
        j = tf.matrix_diag(tf.matrix_transpose(1. / self.X_var)) + Koo  # q x n x n
        Lj = tf.cholesky(j)
        mu_xn = tf.matmul(Kno, self.X_mean)  # n x q
        temp3 = tf.matrix_triangular_solve(Lj, tf.tile(tf.expand_dims(tf.transpose(Kno), 0),
                                                       [self.num_latent, 1, 1]))  # q x n x n
        var_xn = Knn - tf.einsum('ijk,ijl->ikl', temp3, temp3)
        var_xn = tf.transpose(tf.matrix_diag_part(var_xn))
        return mu_xn, var_xn