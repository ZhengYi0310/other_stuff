import gpflow
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
N = 12
X = np.random.rand(N, 1)
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3


meanf = gpflow.mean_functions.Linear(1,0)
m = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(1, lengthscales=0.3), meanf)
m.likelihood.variance = 0.01

m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
m.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
m.likelihood.variance.prior = gpflow.priors.Gamma(1., 1.)
m.mean_function.A.prior = gpflow.priors.Gaussian(0., 10.)
m.mean_function.b.prior = gpflow.priors.Gaussian(0., 10.)

samples = m.sample(500, epsilon=0.2, verbose=1)
def plot(m):
    xx = np.linspace(-0.1, 1.1, 100)[:, None]
    mean, var = m.predict_y(xx)
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2 * np.sqrt(var[:,0]), mean[:,0] +
                     2 * np.sqrt(var[:,0]), color = 'blue', alpha=0.2)
sample_df = m.get_samples_df(samples)
print sample_df.head()
# plt.figure(figsize=(16, 4))
# for lab, s in sample_df.iteritems():
#     plt.plot(s, label=lab)
#     plt.legend(loc=0)
#plot the function posterior
xx = np.linspace(-0.1, 1.1, 100)[:,None]
plt.figure(figsize=(12, 6))
for i, s in sample_df.iterrows():
    m.set_parameter_dict(s)
    f = m.predict_f_samples(xx, 1)
    plt.plot(xx, f[0,:,:], 'b', lw=2, alpha = 0.05)

plt.plot(X, Y, 'kx', mew=2)
plt.xlim(xx.min(), xx.max())
plt.ylim(0, 6)
plt.show()


f, axis = plt.subplots(1, 3, figsize=(12,4))
axis[0].plot(sample_df['name.likelihood.variance'],
             sample_df['name.kern.variance'],
             'k',
             alpha=0.15)
axis[0].set_xlabel('noise_variance')
axis[0].set_ylabel('signal_variance')

axis[1].plot(sample_df['name.likelihood.variance'],
             sample_df['name.kern.lengthscales'],
             'k',
             alpha=0.15)
axis[1].set_xlabel('noise_variance')
axis[1].set_ylabel('lengthscale')

axis[2].plot(sample_df['name.kern.lengthscales'],
            sample_df['name.kern.variance'], 'k.', alpha = 0.1)
axis[2].set_xlabel('lengthscale')
axis[2].set_ylabel('signal_variance')

