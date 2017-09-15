#!/usr/bin/env python

'''
Implementation of VAE with dynamics: 
http://www.ausy.tu-darmstadt.de/uploads/Site/EditPublication/hoof2016IROS.pdf

Author:Yi Zheng
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as contrib
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import ipdb as pdb
import seaborn as sns

sns.set_style("whitegrid")

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/bml/yi_zheng/data', 'Directory for training data.')
flags.DEFINE_string('log_dir' , '/home/bml/yi_zheng/log', 'Directory for log files.')
flags.DEFINE_integer('latent_dim', 5, 'Dimensionality for the latent state.')
flags.DEFINE_integer('input_dim', 12 * 19, 'Dimensionality for the input')
flags.DEFINE_integer('control_dim', 7, 'Dimensionality for the control signal')
flags.DEFINE_integer('batch_size', 100, 'Minibatch size.')
flags.DEFINE_integer('n_samples', 20, 'Number of samples per-data point of X.')
flags.DEFINE_integer('print_every', 1000, 'Print every n iterations.')
flags.DEFINE_integer('n_iterations', '1000000', 'number of iterations')
FLAGS = flags.FLAGS

sg = tf.contrib.bayesflow.stochastic_tensor
distributions = contrib.distributions

# class NormalDistribution(object):
#     '''
#     Represents a multivariate normal distribution parameterized by
#     N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
#     Cov=A*(sigma).^2*A', where A = (I+v*r^T).
#     '''


def Linear(x, output_dim):
    W = tf.get_variable('W', [x.get_shape()[1], output_dim], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b

def transition_dynamics(z, u, latent_dim):
    W_z = tf.get_variable('W_z', [z.get_shape()[1], latent_dim], initializer=tf.constant_initializer(0.0))
    W_u = tf.get_variable('W_u', [FLAGS.control_dim, latent_dim], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('b', [latent_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(z, W_z) + tf.matmul(u, W_u) + b

def ReLU(x, output_dim, scope):
    with tf.variable_scope(scope):
        return tf.nn.relu(Linear(x, output_dim))

def encoder_network(x, latent_dim, share=None):
    '''
        Construct an encoder network parametrizing a Gaussian.

        :param x: A batch of inputs (biotac_data). 
        :param latent_dim: The dimensions for the latent states.
        :return: mu: Mean parameters for the Normal distribution variational family.
                 sigma: Standard deviation parameters for the Normal distribution variational family.
    '''
    with tf.variable_scope("encoder_net", reuse=share):
        net = slim.flatten(x)
        net = ReLU(net , 512, "layer" + str(1)) # According to the Van Hoof paper, only one hidden layer with 512 nodes is chosen
        gaussian_params = Linear(net, latent_dim * 2)
        mu = gaussian_params[:, :latent_dim]
        sigma = gaussian_params[:, latent_dim:]
        return mu, sigma

def decoder_network(z, input_dim, share=None):
    '''
       Construct an decoder network, for non-binary data parametrizing a Gaussian.
        
       :param z: A batch of latent features. 
       :param share: 
       :return: mu: Mean parameters for the Normal distribution variational family.
                sigma: Standard deviation parameters for the Normal distribution variational family.
    '''
    with tf.variable_scope("decoder_net", reuse=share):
        net = ReLU(z, 512, "layer" + str(1)) # According to the Van Hoof paper, only one hidden layer with 512 nodes is chosen
        gaussian_params = Linear(net, input_dim * 2)
        mu = gaussian_params[:, :input_dim]
        sigma = gaussian_params[:, input_dim:]
        return mu, sigma

def sample_Q_phi(mu, sigma, share=False):
    with tf.variable_scope("sampleQ_phi", reuse=share):
        stddrv = tf.sqrt(tf.exp(sigma))
        return distributions.MultivariateNormalDiag(mu, stddrv).sample(FLAGS.n_samples)

def InfoLost(Q_phi):
    '''

    :param Q_phi: the variational distribution given the input
    :return: the KL divergence between the variation distribution give the input and the standard normal prior, shape 1 by batch
    '''
    prior_z = distributions.MultivariateNormalDiag(np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                                   np.ones(FLAGS.latent_dim, dtype=np.float32))
    return distributions.kl(Q_phi, prior_z)

# Build the networks
x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_dim])
u = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.control_dim]) # control at time step t
x_next = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_dim]) # control at time step t+1


# encoder x at time stamp t
mu, sigma = encoder_network(x, FLAGS.latent_dim)
# sample the latent feature
z_samples = sample_Q_phi(mu, sigma)
# put samples generated from one x input together
z_samples = tf.split(z_samples, num_or_size_splits=FLAGS.batch_size, axis=1)
# compute the z sample for the next time stamp based on z sample for the previous time stamp
z_samples_next = tf.zeros(20 * FLAGS.batch_size)
recon_loss = []
for i in range(0, FLAGS.batch_size):
    # squeeze the dimension of the sample batch for one x input
    with tf.variable_scope("transition_dynamics") as scope:
        if (i > 0):
            scope.reuse_variables()
        temp_z_samples_next = transition_dynamics(tf.squeeze(z_samples[i], axis=1), tf.reshape(u[i], [-1, FLAGS.control_dim]), FLAGS.latent_dim)
        # print(temp_z_samples_next)

    # for each batch of x input, decode the n_samples of z samples
    x_mu, x_sigma = decoder_network(temp_z_samples_next, FLAGS.input_dim, True if i > 0 else None)
    print(x_sigma)
    # compute the posterior predictive probability for the x input of the next time stamp
    expected_log_likelihood = distributions.MultivariateNormalDiag(x_mu, x_sigma).prob(tf.reshape(x_next[i], [-1, FLAGS.input_dim]))
    recon_loss.append(tf.reduce_sum(expected_log_likelihood) / FLAGS.n_samples)
# compute the kl divergence
with tf.variable_scope("loss"):
    info_loss = InfoLost(distributions.MultivariateNormalDiag(mu, sigma))
    recon_loss = tf.reshape(recon_loss, [-1, FLAGS.batch_size])
    total_loss = tf.reduce_sum(info_loss + recon_loss)
print recon_loss
print info_loss
print total_loss
for v in tf.all_variables():
    print("%s : %s" % (v.name, v.get_shape()))

pdb.set_trace()

# construct the cost
with tf.variable_scope("Optimizer"):
    alpha = 1e-4
    beta2 = 0.1
    optimizer = tf.train.AdamOptimizer(alpha,  beta2=beta2)  # beta2=0.1
    train_op = optimizer.minimize(total_loss)

saver = tf.train.Saver() # saves variables learned during training
# summaries
tf.summary.scalar("total_loss", total_loss)
tf.summary.scalar("recon_loss", tf.reduce_mean(recon_loss))
tf.summary.scalar("info_loss", tf.reduce_mean(info_loss))
tf.summary.merge_all()


mvn = distributions.MultivariateNormalDiag([[1., 2, 3], [11, 22, 33]] ,
                                [[1., 2, 3],[0.001, 0.001, 0.001]])  # shape: [2, 3]
sess = tf.InteractiveSession()
sample = mvn.sample(3, seed=0)
a = sample.eval()
print(a)
b = tf.split(a, num_or_size_splits=2, axis=1)
print(b[0].eval())
split0, split1 = tf.split(a, num_or_size_splits=2, axis=1)
split0 = tf.squeeze(split0, axis=1)
print(split0.eval())






