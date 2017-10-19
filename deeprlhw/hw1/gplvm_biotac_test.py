from __future__ import print_function
import gpflow
from gpflow import ekernels
from gpflow import kernels
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import gplvm
import pods

gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations
# First load the dataset
my_data1 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_0.txt', delimiter=',')[1:, 25:44]
my_data2 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_11.txt', delimiter=',')[1:, 25:44]
my_data3 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_6.txt', delimiter=',')[1:, 25:44]
base_line = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_1.txt', delimiter=',')[1:, 25:44]
my_data1 = my_data1 - base_line
my_data2 = my_data2 - base_line
my_data3 = my_data3 - base_line
my_data1 = (my_data1 - np.min(my_data1, axis=0)) / (np.max(my_data1 ,axis=0) - np.min(my_data1, axis=0))
my_data2 = (my_data2 - np.min(my_data2, axis=0))/ (np.max(my_data2 ,axis=0) - np.min(my_data2, axis=0))
my_data3 = (my_data3 - np.min(my_data3, axis=0))/ (np.max(my_data3 ,axis=0) - np.min(my_data2, axis=0))
elec_data = np.vstack((my_data1, my_data2, my_data3))
# labels = np.vstack((np.zeros((my_data1.shape[0], 1), dtype=int), np.ones((my_data2.shape[0], 1), dtype=int), np.ones((my_data2.shape[0], 1), dtype=int) * 2))
# labels = labels.reshape(-1)


data = pods.datasets.oil_100()
Y = data['X']

labels = np.argmax(data['Y'], axis=1)
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
print(labels)
Q = 10
N = Y.shape[0]
M = 30  # number of inducing pts
X_mean = gplvm.PCA_initialization(Y, Q) # Initialise via PCA
Z = np.random.permutation(X_mean.copy())[:M]



fHmmm = False
if(fHmmm):
    k = ekernels.Add([ekernels.RBF(3, ARD=True, active_dims=slice(0,3)),
                  ekernels.Linear(2, ARD=False, active_dims=slice(3,5))])
else:
    k = ekernels.RBF(Q, ARD=True)

m = gplvm.BayesianGPLVM(X_variational_mean=X_mean, X_variational_var=0.1*np.ones((N, Q)), Y=Y,
                                Kern=k, M=M, Z=Z)
m.likelihood.variance = 0.01
m.optimize(disp=True, maxiter=500)

kern = m.kern
sens = np.sqrt(kern.variance.value) / kern.lengthscales.value
print(m.kern)
print(sens)
fig, ax = plt.subplots()
ax.bar(np.arange(len(kern.lengthscales.value)) , sens, 0.1, color='y')
ax.set_title('Sensitivity to latent inputs')

XPCAplot = gpflow.gplvm.PCA_reduce(Y, Q)
fig = plt.figure(figsize=(10, 6))



ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for i, c in zip(np.unique(labels), colors):
    ax1.scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], XPCAplot[labels==i,2], color=c, label=i)
    ax1.set_title('PCA')
    ax1.legend()
    ax2.scatter(m.X_variational_mean.value[labels==i,0], m.X_variational_mean.value[labels==i,1], m.X_variational_mean.value[labels==i,2], color=c, label=i)
    ax2.set_title('Bayesian GPLVM')
    ax2.legend()
plt.show()