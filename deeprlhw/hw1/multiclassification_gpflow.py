from __future__ import print_function
import gpflow
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# make a one dimensional classification problem
np.random.seed(1)
X = np.random.rand(100, 1)
K = np.exp(-0.5 * np.square(X - X.T) / 0.01) + np.eye(100) * 1e-6
f = np.dot(np.linalg.cholesky(K), np.random.randn(100, 3))


plt.figure(figsize=(12,6))
plt.plot(X, f, '.')
plt.show()
Y = np.argmax(f, 1).reshape(-1,1).astype(float)

m = gpflow.svgp.SVGP(X, Y, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.White(1, variance=0.01), likelihood=gpflow.likelihoods.MultiClass(3),
                     Z=X[::5].copy(), num_latent=3, whiten=True, q_diag=True)
m.kern.white.variance.fixed = True
m.Z.fixed = True
_ = m.optimize()

def plot(m):
    f = plt.figure(figsize=(12, 6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

    xx = np.linspace(m.X.value.min(), m.X.value.max(), 200).reshape(-1, 1)
    mu, var = m.predict_f(xx)
    mu, var = mu.copy(), var.copy()
    p, _ = m.predict_y(xx)

    a3.set_xticks([])
    a3.set_yticks([])

    for i in range(m.likelihood.num_classes):
        x = m.X.value[m.Y.value.flatten()==i]
        points, = a3.plot(x, x*0, '.')
        color = points.get_color()
        a1.plot(xx, mu[:,i], color=color, lw=2)
        a1.plot(xx, mu[:,i] + 2 * np.sqrt(var[:, i]), '--', color=color)
        a1.plot(xx, mu[:,i] - 2 * np.sqrt(var[:, i]), '--', color=color)
        a2.plot(xx, p[:,i], '-', color=color, lw=2)

    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])

# Sparse MCMC
m = gpflow.sgpmc.SGPMC(X ,Y,
                       kern=gpflow.kernels.Matern32(1, lengthscales=0.1) + gpflow.kernels.White(1, variance=0.01),
                       likelihood=gpflow.likelihoods.MultiClass(3),
                       Z=X[::5].copy(),
                       num_latent=3)
m.kern.matern32.variance.prior = gpflow.priors.Gamma(1., 1.)
m.kern.matern32.lengthscales.prior = gpflow.priors.Gamma(2., 2.)
m.kern.white.variance.fixed = True
m.optimize(maxiter=10)

samples = m.sample(500, verbose=True, epsilon=0.04, Lmax=15)

def plot_from_samples(m, samples):
    f = plt.figure(figsize=(12, 6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9 , 0.1])

    xx = np.linspace(m.X.value.min(), m.X.value.max(), 200).reshape(-1, 1)

    Fpred, Ypred = [], []
    for s in samples[100::10]:
        m.set_state(s)
        Ypred.append(m.predict_y(xx)[0])
        Fpred.append(m.predict_f_samples(xx, 1).squeeze())

    for i in range(m.likelihood.num_classes):
        x = m.X.value[m.Y.value.flatten()==i]
        points, = a3.plot(x, x*0, '.')
        color = points.get_color()
        for F in Fpred:
            a1.plot(xx, F[:, i], color=color, lw=0.2, alpha=1.0)
        for Y in Ypred:
            a2.plot(xx, Y[:, i], color=color, lw=0.5, alpha=1.0)

    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])

    a3.set_xticks([])
    a3.set_yticks([])
plot_from_samples(m, samples)
plt.show()
