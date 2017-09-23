import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class SPGPs(object):
    '''
    Edward Snelson, Zoubin, Ghahramani, Sparse Pseudo-input Gaussian Processes, 2005

    X: training inputs (N * D)
    y: training outputs (N * 1)
    M: the number of pseudo-inputs
    theta: all parameters for the SPGPs model, including the pseudo inputs and hyper parameters
           for the ARD kernel
    hyp_params: hyper parameters for the ARD kernel


           theta = np.array([np.reshape(x_pseudo, -1, M * D), hyp_params])
           hyp_params = np.array([])  (D + 2) * 1
              hyp_params[:, :D] = log(b)
              hyp_params[:, D] = log(c)
              hyp_params[:, D + 1] = log(sig)

    jitter -- optional jitter (default 1e-6)
    '''
    if __name__ == '__main__':
        def __init__(self, theta, y, X, M, jitter):
            self.X = X
            self.y = y
            self.M = M
            self.jitter = jitter

            self.N, self.D = X.shape
            assert y.shape[0] == self.N, "the number of training inputs and targets are not the same!"
            assert y.shape[1] == 1, "training targets dimension should be 1!"

            temp = self.M * self.D
            self.x_pseudo =np.reshape(theta[:, :temp], (self.M ,self.D))
            self.b = np.exp(theta[:, temp:temp + self.D])
            self.c = np.exp(theta[:, temp + self.D])
            self.sigma = np.exp(theta, temp + self.D + 1)

            # Compute the kernel matrix
            X = self.X * np.sqrt(self.b)
            x_pseudo = self.x_pseudo * np.sqrt(self.b)
            # The kernel matrix for the pseudo-inputs
            self.K_mm = self.compute_kernel_mat(x_pseudo, x_pseudo.transpose())
            self.K_mm = self.c * (np.exp(-0.5 * self.K_mm)) + self.sigma * np.identity(self.M)
            # The kernel matrix for the training inputs
            self.K_xx = self.compute_kernel_mat(X, X.transpose())
            self.K_xx = self.c * (np.exp(-0.5 * self.K_xx)) + self.sigma * np.identity(self.N)
            # The kernel matrix between the training points and the pseudo inputs
            

    def compute_kernel_mat(self, X1, X2):
        kernel = np.dot(X1, X2.transpose())
        kernel = 2 * np.diag(np.diag(kernel)) - 2 * kernel
        return np.exp(-0.5 * kernel)


a = np.array([[1, 2]])
b = np.array([[3]])
d = np.hstack((a ,b))
print d
print b