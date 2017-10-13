import gpflow
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as contrib

# def plotkernelsample(k, ax, xmin=-3, xmax=3):
#     xx = np.linspace(xmin, xmax, 100)[:,None]
#     K = k.compute_K_symm(xx)
#     ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
#     ax.set_title(k.__class__.__name__)
#
# def plotkernelfunction(k, ax, xmin=-3, xmax=3, other=0):
#     xx = np.linspace(xmin, xmax, 100)[:, None]
#     K = k.compute_K_symm(xx)
#     ax.plot(xx, k.compute_K(xx, np.zeros((1, 1)) + other))
#     ax.set_title(k.__class__.__name__ + ' k(x, %f) '%other)
#
# f, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
# plotkernelsample(gpflow.kernels.Matern12(1), axes[0,0])
# plotkernelsample(gpflow.kernels.Matern32(1), axes[0,1])
# plotkernelsample(gpflow.kernels.Matern52(1), axes[0,2])
# plotkernelsample(gpflow.kernels.RBF(1), axes[0,3])
# plotkernelsample(gpflow.kernels.Constant(1), axes[1,0])
# plotkernelsample(gpflow.kernels.Linear(1), axes[1,1])
# plotkernelsample(gpflow.kernels.Cosine(1), axes[1,2])
# plotkernelsample(gpflow.kernels.PeriodicKernel(1), axes[1,3])
# axes[0,0].set_ylim(-3, 3)
# plt.show()
sess = tf.InteractiveSession()
mat_r = tf.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
mat_r = tf.cast(mat_r, tf.float64)
mat_l = tf.convert_to_tensor(np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
L = tf.cholesky(mat_l)
print(sess.run(L))
# print(sess.run((mat_l / mat_r)))
# tmp = tf.matrix_triangular_solve(L, mat_r, lower=True)
# print(sess.run(tmp))
# print(sess.run(tf.matmul(tf.matrix_inverse(L), mat_r)))
# result1 = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True)
# print(sess.run(tf.size(result1)))
# result2 = tf.matrix_triangular_solve(tf.transpose(L), tmp, lower=False)
# print(sess.run(tf.size(result2)))
# result3 = tf.matmul(tf.matrix_inverse(mat_l), mat_r)
# result4 = tf.matmul(tf.matrix_inverse(tf.transpose(L)), tf.matmul(tf.matrix_inverse(L), mat_r))
# print(sess.run(result3))
# print(sess.run(result4))
sess.close()

# print (mat_l.get_shape().ndims)
# a = np.array([[1.0, 2.0, 3.0],[2.0, 8.0, 8.0], [3.0, 8.0, 35.0]])
# b = np.array([[1, 2, 3]])
# #assert (a.shape == b.shape), 'asdsad {}'.format(2)
# print np.cov(a.T)
# evecs, evals = np.linalg.eigh(np.cov(a.T))
# print evals
# i = np.argsort(evecs)[::-1]
# W = evals[:, i]
# print W