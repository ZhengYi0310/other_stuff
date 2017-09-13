import itertools
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.misc import imsave
from tensorflow.examples.tutorials import mnist

sns.set_style("whitegrid")
sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.contrib.distributions

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/bml/yi_zheng/data', 'Directory for training data.')
flags.DEFINE_string('log_dir' , '/home/bml/yi_zheng/log', 'Directory for log files.')
flags.DEFINE_integer('latent_dim', 5, 'Dimensionality for the latent state.')
flags.DEFINE_integer('batch_size', 100, 'Minibatch size.')
flags.DEFINE_integer('n_samples', 1, 'Number of samples per-data point of X.')
flags.DEFINE_integer('print_every', 1000, 'Print every n iterations.')
flags.DEFINE_integer('n_iterations', '1000000', 'number of iterations')

FLAGS = flags.FLAGS

def encoder_network(x, latent_dim):
    '''
    Construct an encoder network parametrizing a Gaussian.
    
    :param x: A batch of MNIST digits. 
    :param latent_dim: The dimensions for the latent states.
    :return: mu: Mean parameters for the Normal distribution variational family.
             sigma: Standard deviation parameters for the Normal distribution variational family.
    '''
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.flatten(x)
        net = slim.fully_connected(net, 800)
        net = slim.fully_connected(net, 512)
        gaussian_params = slim.fully_connected(net, latent_dim * 2, activation_fn=None)
        mu = gaussian_params[:, :latent_dim]
        sigma = tf.exp(gaussian_params[:, latent_dim:])
        return mu, sigma

def decoder_network(z, input_dim):
    '''
    For non-binary data, construct an encoder network parametrizing a Gaussian.
    :param z: Samples of latent variables 
    :return: mu: Mean parameters for the Normal distribution variational family.
             sigma: Standard deviation parameters for the Normal distribution variational family.
    '''
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.fully_connected(z, 512)
        net = slim.fully_connected(net, 800)
        gaussian_params = slim.fully_connected(net, input_dim * 2, activation_fn=None)
        mu = gaussian_params[:, :input_dim]
        sigma = tf.exp(gaussian_params[:, input_dim:])
        return mu, sigma

def train():
    # Train a vairational Autoencoder on the Biotac data
    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, [None, 12, 19, 1])
        tf.summary.tensor_summary('x', x)

    with tf.variable_scope('variational'):
        q_mu, q_sigma = encoder_network(x=x, latent_dim=FLAGS.latent_dim)
        with st.value(st.SampleValue()):
            # The variational distribution is a Normal with mean and standard
            # deviation given by the encoder network
            q_z = st.StochasticTensor(distributions.MultivariateNormalDiag(loc=q_mu, scale=tf.sqrt(q_sigma)))

    with tf.variable_scope('model'):
        p_mu, p_sigma = decoder_network(z=q_z, input_dim = 12 * 19)
        with st.value_type(st.SampleValue()):
            # The likelihood distribution is a Normal with mean and standard
            # deviation given by the decoder network
            p_x_given_z_posterior = st.StochasticTensor(distributions.MultivariateNormalDiag(loc=p_mu, scale=tf.sqrt(p_sigma)))
            tf.summary.tensor_summary('p_x_given_z_posterior',
                             tf.cast(p_x_given_z_posterior, tf.float64))

    # Take samples from the prior
    with tf.variable_scope('model', reuse=True):
        p_z = distributions.MultivariateNormalDiag(loc=np.zeros(FLAGS.latent_dim, dtype=np.float64),
                                   scale=np.ones(FLAGS.latent_dim, dtype=np.float64))
        p_z_sample = p_z.sample(FLAGS.n_samples)
        p_mu, p_sigma = decoder_network(z=p_z_sample, input_dim=12 * 19)
        with st.value_type(st.SampleValue()):
            # The likelihood distribution is a Normal with mean and standard
            # deviation given by the decoder network
            p_x_given_z_prior = st.StochasticTensor(distributions.MultivariateNormalDiag(loc=p_mu, scale=tf.sqrt(p_sigma)))
            tf.summary.tensor_summary('p_x_given_z_prior', tf.cast(p_x_given_z_prior, tf.float64))

    # Take samples from the prior using a placeholder
    with tf.variable_scope('mode', reuse=True):
        z_input = tf.placeholder(tf.float64, [None, FLAGS.latent_dim])
        p_mu, p_sigma = decoder_network(z=z_input, input_dim=12 * 19)
        with st.value_type(st.SampleValue()):
            # The likelihood distribution is a Normal with mean and standard
            # deviation given by the decoder network
            p_x_given_z_prior_inp = st.StochasticTensor(distributions.MultivariateNormalDiag(loc=p_mu, scale=tf.sqrt(p_sigma)))
            tf.summary.tensor_summary('p_x_given_z_prior', tf.cast(p_x_given_z_prior_inp, tf.float64))

    # Build the variational lower bound
    kl = distributions.kl_divergence(q_z.distribution, p_z)
    expected_log_likelihood = tf.reduce_sum(p_x_given_z_posterior.distribution.log_prob(x))
    elbo = tf.reduce_sum(expected_log_likelihood - kl)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(-elbo)

    # Merge all the summaries
    summary_op = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()

    # Run training
    sess = tf.interactiveSession()
    sess.run(init_op)

    mnist_set = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
    print('Saving TensorBoard summaries and images to :%s' %FLAGS.logdir)
    train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    # Get fixed MNIST digits for plotting posterior means during training
    np_x_fixed, np_y = mnist_set.test.next_batch(5000)
    np_x_fixed = np_x_fixed.reshape(5000, 28, 28, 1)
    np_x_fixed = (np_x_fixed > 0.5).astype(np.float32)

    for i in range(FLAGS.n_iterations):
        np_x, _ = mnist_set.train.next_batch(FLAGS.batch_size)
        np_x = np_x.reshape(FLAGS.batch_size, 28, 28, 1)
        sess.run(train_op, {x: np_x})

        # Print progress and save images every so often
        t0 = time.time()
        if ((i % FLAGS.print_every) == 0):
            np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
            train_writer.add_summary(summary_str, i)
            print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(
                    i,
                    np_elbo / FLAGS.batch_size,
                    FLAGS.batch_size * FLAGS.print_every / (time.time() - t0)))
            t0 = time.time()

            # Save samples
            p_x_given_z_posterior_samples, p_x_given_z_prior_samples = sess.run([p_x_given_z_posterior, p_x_given_z_prior], {x: np_x})
            for k in range(FLAGS.n_samples):
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_posterior_predictive_%d_data.jpg' % (i, k))
                imsave(f_name, np_x[k, :, :, 0])
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_posterior_predictive_%d_sample.jpg' % (i, k))
                imsave(f_name, p_x_given_z_posterior_samples[k, :, :, 0])
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_prior_predictive_%d.jpg' % (i, k))
                imsave(f_name, p_x_given_z_prior_samples[k, :, :, 0])

            # Plot the posterior predictive space
            if FLAGS.latent_dim == 3:
                np_q_mu = sess.run(q_mu, {x: np_x_fixed})



def sampleNormal(mu, sigma):
    # note: sigma is diagonal standard deviation, not variance
    n01 = tf.random_normal(mu.get_shape(), mean=0, stddev=1)
    return mu + sigma * n01

def sample_Q_phi(mu, sigma):
    """
      Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
      mu is (batch,z_size)
      
      # note: sigma here is the diagonal vector of the covariance matrix 
    """
    with tf.variable_scope("sample_Q_phi"):
        with tf.variable_scope("Q_phi"):
            return sampleNormal(mu, tf.sqrt(sigma))

def sample_P_theta(mu, sigma):
    """
          Samples Xt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
          mu is (batch,z_size)

          # note: sigma here is the diagonal vector of the covariance matrix 
        """
    with tf.variable_scope("sample_P_theta"):
        with tf.variable_scope("P_theta"):
            return sampleNormal(mu, tf.sqrt(sigma))
