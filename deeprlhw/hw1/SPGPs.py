import numpy as np
import scipy.linalg as slg
import ARD


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

        return indent(self.__class__.__name__ + '(', '\n'.join([indent(indent('likehood=', repr(self.likelihood_)),
                                                                       indent('kernel=', repr(self.kernel_)),
                                                                       indent('mean=', str(self.mean_)))]) + ')')

    def __param__(self):
        params = []
        params.append([('kernel.%s' % p[0] + p[1:] for p in self.kernel_.__param__())])
        params.append([('likelihood.%s' % p[0] + p[1:] for p in self.kernel_.__param__())])
        params.append([('mean, 1, Falsue')])
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


m = np.array([[2,0], [0,2]])
print slg.cholesky(m)
