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
from vgpds_test import BayesianDGPLVM, TimeSeries
import gpflow

gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations
# First load the dataset
my_data1 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_0.txt', delimiter=',')[1:, 25:44]
t_data1 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_0.txt', delimiter=',')[1:, 44]
t_data1 = np.round(t_data1 - t_data1[0], decimals=2)
t_data1 = np.reshape(t_data1, (t_data1.shape[0], 1))

my_data2 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_3.txt', delimiter=',')[1:, 25:44]
t_data2 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_6.txt', delimiter=',')[1:151, 44]
t_data2 = np.round(t_data2 - t_data2[0], decimals=2)
t_data2 = np.reshape(t_data2, (t_data2.shape[0], 1))

my_data3 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_6.txt', delimiter=',')[1:, 25:44]
t_data3 = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_11.txt', delimiter=',')[1:, 44]
t_data3 = np.round(t_data3 - t_data3[0], decimals=2)
t_data3 = np.reshape(t_data3, (t_data3.shape[0], 1))



base_line = np.genfromtxt(os.path.dirname(__file__) + '/data/biotac_sensors__trial_9.txt', delimiter=',')[1:, 25:44]
print(np.mean(base_line, axis=0))
my_data1 = my_data1
my_data2 = my_data2
my_data3 = my_data3

# elec_data = np.vstack((my_data1, my_data3, my_data2))
# my_data1 = (my_data1 - np.min(my_data1, axis=0)) / (np.max(my_data1 ,axis=0) - np.min(my_data1, axis=0))
# my_data2 = (my_data2 - np.min(my_data2, axis=0))/ (np.max(my_data2 ,axis=0) - np.min(my_data2, axis=0))
# my_data3 = (my_data3 - np.min(my_data3, axis=0))/ (np.max(my_data3 ,axis=0) - np.min(my_data3, axis=0))
# elec_data = np.concatenate((np.concatenate((my_data1, my_data2), axis=0), my_data3), axis=0)


elec_data = np.vstack((my_data1, my_data3, my_data2))
t_elec_data = [t_data1, t_data2, t_data3]
labels = np.vstack((np.zeros((my_data1.shape[0], 1), dtype=int), np.ones((my_data2.shape[0], 1), dtype=int), np.ones((my_data2.shape[0], 1), dtype=int) * 2))
labels = labels.reshape(-1)


data = pods.datasets.oil_100()
Y = data['X']
#
# labels = np.argmax(data['Y'], axis=1)
# labels= data['Y'].argmax(axis=1)
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
# print(labels)
Q = 10
# N = my_data3.shape[0]

N = elec_data.shape[0]
# N = Y.shape[0]
M = 30  # number of inducing pts
# X_mean_data1 = gplvm.PCA_initialization(my_data1, Q) # Initialise via PCA
# X_mean_data2 = gplvm.PCA_initialization(my_data2, Q)
# X_mean_data3 = gplvm.PCA_initialization(my_data3, Q)
# X_mean = [X_mean_data1, X_mean_data2, X_mean_data3]
# X_var=[0.1*np.ones((N, Q))] * len(X_mean)

# X_mean = gplvm.PCA_initialization(Y, Q) # Initialise via PCA
X_mean = gplvm.PCA_initialization(elec_data, Q) # Initialise via PCA
X_var = 0.1*np.ones((N, Q))
Z = np.random.permutation(X_mean.copy())[:M]



#
# #
# #
# #
# fHmmm = False
# if(fHmmm):
#     k = ekernels.Add([ekernels.RBF(3, ARD=True, active_dims=slice(0,3)),
#                   ekernels.Linear(2, ARD=False, active_dims=slice(3,5))])
# else:
#     k = ekernels.RBF(Q, ARD=True)
# kt = gpflow.ekernels.RBF(1)
#
# m = BayesianDGPLVM(X_mean=X_mean, X_var=X_var, Y=elec_data,
#                                 kern=k, time=t_elec_data, kern_t=kt, M=M)
# # m = gplvm.BayesianGPLVM(X_variational_mean=X_mean, X_variational_var=X_var, Y=elec_data,
# #                                  Kern=k, M=M)
# m.likelihood.variance = 0.1
# m.optimize(disp=True, maxiter=500)
#
# kern = m.kern
# sens = np.sqrt(kern.variance.value) / kern.lengthscales.value
# print(m.kern)
# print(sens)
# fig, ax = plt.subplots()
# ax.bar(np.arange(len(kern.lengthscales.value)) , sens, 0.1, color='y')
# ax.set_title('Sensitivity to latent inputs')
# plt.show()


XPCAplot = gplvm.PCA_initialization(elec_data, Q)
# XPCAplot = gpflow.gplvm.PCA_reduce(Y, Q)
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for i, c in zip(np.unique(labels), colors):
    ax1.scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], XPCAplot[labels==i,2], color=c, label=i)
    ax1.set_title('PCA')
    ax1.legend()
    # ax2.scatter(m.series[i].X_mean.value[:,0], m.series[i].X_mean.value[:,1], m.series[i].X_mean.value[:,2], color=c, label=i)
    # ax2.scatter(m.X_variational_mean.value[labels==i, 0], m.X_variational_mean.value[labels==i, 1], m.X_variational_mean.value[labels==i, 2], color=c, label=i)
    # ax2.set_title('Bayesian GPLVM')
    # ax2.legend()
plt.show()





# np.random.seed(2)
#
# N = 100
# kt = gpflow.ekernels.RBF(1)
# kx = gpflow.ekernels.RBF(1)
#
# #
# t = np.arange(N)
# t = np.reshape(t, (N, 1)) * .2
# X = np.random.multivariate_normal(np.zeros(N), kt.compute_K(t, t), 1).T
# Y = np.random.multivariate_normal(np.zeros(N), kx.compute_K(X, X), 2).T
#
#
#
# X_mean = gpflow.gplvm.PCA_reduce(Y, 1)
# X_mean = 0.01*(X_mean-np.mean(X_mean, axis=0))/np.std(X_mean, axis=0)
# X_var = np.ones((N, 1))
#
# # print([X_mean][0])
# # print([X_var][0])
# # print([t][0])
# # m = BayesianDGPLVM(X_variational_mean=[X_mean], X_variational_var=[X_var], Y=Y, kern=kx, t=[t],
# #                    kern_t=kt, M=6)
# m = BayesianDGPLVM(X_mean=[X_mean], X_var=[X_var], Y=Y, kern=kx, time=[t],
#                    kern_t=kt, M=6)
# m.optimize(maxiter=50000, disp=True)
#
# #prediction
# N = 500
# tn = np.reshape(np.linspace(0, np.max(t) * 3, N), (N, 1))
# new, var = m.predict_serie(tn)
#
# # print(new[:, 0])
# ax = plt.subplot(1, 1, 1)
# ax.plot(tn, new[:, 0] + 2 * np.sqrt(var[:, 0]), 'b--')
# ax.plot(tn, new[:, 0] - 2 * np.sqrt(var[:, 0]), 'b--')
# ax.plot(tn, new[:, 0], 'b')
# ax.scatter(t[:, 0], Y[:, 0], c='b')
# ax.plot(tn, new[:, 1] + 2 * np.sqrt(var[:, 1]), 'g--')
# ax.plot(tn, new[:, 1] - 2 * np.sqrt(var[:, 1]), 'g--')
# ax.plot(tn, new[:, 1], 'g')
# ax.scatter(t[:, 0], Y[:, 1], c='g')
# ax.set_xlabel('t')
# ax.set_ylabel('f(t)')
# ax.set_ylim(-3, 3)
# plt.show()