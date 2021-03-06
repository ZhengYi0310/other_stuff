import numpy as np
import scipy.linalg as slg
from ARD import ARD
from likelihood import Gaussianlikelihood


class GP(object):
    '''
    Gaussian Process inference model

    This class define the basic interface for the GP inference, the methods that
    must be implemented are:
        '_update': update the internal statistics given all the incoming data
        '_posterior': compute the full posterior probability with sampling
        'posterior': compute the marginal posterior and its gradient.
        'loglikelihood': compute the loglikelihood for the observed data


    Additionally, the following method can be implemented for improved
    performance in some circumstances:
        `_updateinc`: incremental update given new data.
    '''
    def __init__(self, likelihood, kernel, mean):
        self.likelihood_ = likelihood
        self.kernel_ = kernel
        self.mean_ = mean
        self.X_ = None
        self.Y_ = None

        # record the number of hyperparameters, there is an extra +1 due the mean parameter
        self.nhyper_ = self.likelihood_.nhyper_ + self.kernel_.nhyper_ + 1

    def reset(self):
        '''
        Remove all the data from the model
        '''
        self.X_ = None
        self.Y_ = None

    def __repr__(self):
        def indent(pre, text):
            return pre + ('\n' + ' ' * len(pre)).join(text.splitlines())

        return indent(self.__class__.__name__ + '(', '\n'.join([indent('likehood=', repr(self.likelihood_)),
                                                                       indent('kernel=', repr(self.kernel_)),
                                                                       indent('mean=', str(self.mean_))]) + ')')

    def _params(self):
        params = []
        params.append([('kernel.%s' % p[0] + p[1:] for p in self.kernel_._params())])
        params.append([('likelihood.%s' % p[0] + p[1:] for p in self.likelihood_._params())])
        params.append([('mean, 1, False')])
        return params

    def get_hyper(self):
        return np.r_[self.likelihood_.get_hyper(), self.kernel_.get_hyper(), self.mean_]

    def set_hyper(self, hyper):
        num_likelihood_hyper = self.likelihood_.nhyper_
        num_kernel_hyper = self.kernel_.nhyper_

        self.likelihood_.set_hyper(hyper[:num_likelihood_hyper])
        self.kernel_.set_hyper(hyper[num_likelihood_hyper:num_kernel_hyper])
        self.mean_ = hyper[-1]

        if self.ndata > 0:
            self._update()

    def ndata(self):
        """The number of current input/output data pairs."""
        return 0 if (self.X_ is None) else self.X_.shape[0]

    def data(self):
        """The current input/output data pairs"""
        return (self.X_, self.Y_)

    def add_data(self, X, Y):
        X = np.array(X, ndmin=2, dtype=float)
        Y = np.array(Y, ndmin=2, dtype=float)

        if self.X is None:
            self.X = X.copy()
            self.Y = Y.copy()
            self._update()

        else:
            try:
                self._updateinc(X, Y)
                self.X_ = np.r_[self.X_, X]
                self.Y_ = np.r_[self.Y_, Y]

            except NotImplementedError:
                self.X_ = np.r_[self.X_, X]
                self.Y_ = np.r_[self.Y_, Y]
                self._update()

    def sample(self, X, m=None, latent=True, rng=None):
        '''
        Sample from values from posterior given point at 'X'. Give an (n,d) array this
        function will return an n-vector corresponding to such a sample.
        :param X: input values
        :param m: if not None, a (m, n) matrix will be returned, representing m such samples
        :param latent: if False, samples will be corrupted by observartion noise
        :param rng: random seed
        '''
        X = np.array(X, ndmin=2, dtype=float)
        flatten = (m is None)
        m = 1 if flatten else m
        n = X.shape[0]

        # add a tiny amount to the diagonal to make the cholesky of Sigma
        # stable and then add this correlated noise onto mu to get the sample.
        if isinstance(rng, int):
            np.random.seed(rng)
        mu, sigma = self._full_posterior(X)
        f = mu + np.dot(np.random.normal(size=(m, n)), slg.cholesky(sigma)) # depends on the form of returned value of self._full_posterior

        if not latent:
            f = self.likelihood_.sample(f.ravel(), rng).reshape(m, n)

        return f

    def posterior(self, X, grad=False):
        """
        Return the marginal posterior. This should return the mean and variance
                of the given points, and if `grad == True` should return their
                derivatives with respect to the input location as well (i.e. a
                4-tuple).
        """
        return self._marginal_posterior(np.array(X, ndmin=2, dtype=float), grad)

    def _update(self):
        """
        Update any internal parameters (ie sufficient statistics) given the
        entire set of current data.
        """
        raise NotImplementedError

    def _updateinc(self, X, y):
        """
        Update any internal parameters given additional data in the form of
        input/output pairs `X` and `y`. This method is called before data is
        appended to the internal data-store and no subsequent call to `_update`
        is performed.
        """
        raise NotImplementedError

    def _full_posterior(self, X):
        """
        Compute the full posterior at points `X`. Return the mean vector and
        full covariance matrix for the given inputs.
        """
        raise NotImplementedError

    def _marginal_posterior(self, X, grad=False):
        """
        Compute the marginal posterior at points `X`. Return the mean and
        variance vectors for the given inputs. If `grad` is True return the
        gradients with respect to the inputs as well.
        """
        raise NotImplementedError

    def loglikelihood(self, grad=False):
        """
        Return the marginal loglikelihood of the data. If `grad == True` also
        return the gradient with respect to the hyperparameters.
        """
        raise NotImplementedError

class BasicGP(GP):
    """
    Basic Gaussian Process which assumes an ARD kernel and a Gaussian likelihood,
    therefore it can perform exact inference
    """
    def __init__(self, c ,b ,sigma, mu=0, ndim=None):
        likelihood = Gaussianlikelihood(sigma)
        kernel = ARD(c, b, ndim)
        super(BasicGP, self).__init__(likelihood, kernel, mu)
        self.R_ = None
        self.a_ = None

    def reset(self):
        for attr in 'Ra':
            setattr(self, attr + '_', None)
        super(BasicGP, self).reset()

    def _update(self):
        signal_variance = self.likelihood_._variance()
        K = self.kernel_.get(self.X_) + signal_variance * np.eye(len(self.X_))
        r = self.Y_ - self.mean_
        self.R_ = slg.cholesky(K)
        self.a_ = slg.solve_triangular(self.R_, r, trans=True)

    def _update_inc(self, X, y):
        signal_variace = self.likelihood_._variance()
        K_ss = self.kernel_.get(X) + signal_variace * np.eye(len(X))
        K_xs = self.kernel_.get(self.X_, X)
        r = y - self.mean_
        n = self.R_.shape[0]
        m = K_ss.shape[0]
        K_xs = slg.solve_triangular(self.R_, K_xs, trans=True)
        K_ss = slg.cholesky(K_ss - np.dot(K_xs.T, K_xs))
        k_ss = np.dot(K_ss.T, self.a_)

        # grow the new cholesky and then use this to grow the vector a
        self.R_ = np.r_[np.c_[self.R_, K_xs], np.c_[np.zeros(m , n), K_xs]]
        self.a_ = np.r_[self.a_, slg.solve_triangular(K_ss, r - k_ss, trans=True)]

    def _full_posterior(self, X):
        # grab the prior mean and covariance
        mu = np.full(X.shape[0], self.mean_)
        Sigma = self.kernel_.get(X)

        if self.X_ is not None:
            K = self.kernel_.get(self.X_, X)
            V =slg.solve_triangular(self.R_, K, trans=True)
            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(V.t, self.a_)
            Sigma -= np.dot(V.T, V)

        return mu, Sigma

    def _marg_posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self.mean_)
        s2 = self.kernel_.dget(X)

        if self.X_ is not None:
            K = self.kernel_.get(self.X_, X)
            RK = slg.solve_triangular(self.R_, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(RK.T, self.a_)
            s2 -= np.sum(RK ** 2, axis=0)

        if not grad:
            return (mu, s2)

        # Get the prior gradients.
        dmu = np.zeros_like(X)
        ds2 = np.zeros_like(X)

        # NOTE: the above assumes a constant mean and stationary kernel (which
        # we satisfy, but should we change either assumption...).

        if self.X_ is not None:
            dK = self.kernel_.grady(self.X_, X)
            dK = dK.reshape(self.ndata, -1)

            RdK = slg.solve_triangular(self.R_, dK, trans=True)
            dmu += np.dot(RdK.T, self.a_).reshape(X.shape)

            RdK = np.rollaxis(np.reshape(RdK, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(RdK * RK, axis=1).T


m = np.array([[2,0], [0,2]])
print np.reshape(m.ravel() + np.random.normal(0, 0.1, 4), (2, 2))
print slg.cholesky(m)
