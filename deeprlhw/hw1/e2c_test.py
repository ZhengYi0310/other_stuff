import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import tensorflow.contrib.distributions as distributions
from datasets import GymPendulumDatasetV2

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('MNIST_data/', one_hot=True)
n_samples = mnist.train.num_examples

pendulum_dataset = GymPendulumDatasetV2('data/pendulum_markov_train')
n_samples = len(pendulum_dataset)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step=self.global_step,
                                                   decay_steps=10, decay_rate=0.9)

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal(self.z_log_sigma_sq.get_shape(), mean=0, stddev=1)
        # z = mu + sigma*epsilon
        self.z = self.z_mean + tf.exp(self.z_log_sigma_sq) * eps

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean, self.x_reconstr_logsigma_sq = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable("h1_recog", [n_input, n_hidden_recog_1], dtype=tf.float32),
            'h2': tf.get_variable("h2_recog", [n_hidden_recog_1, n_hidden_recog_2], dtype=tf.float32),
            'out_mean': tf.get_variable("out_mean_weights_recog", [n_hidden_recog_2, n_z], dtype=tf.float32),
            'out_log_sigma': tf.get_variable("out_log_sigma_weights_recog", [n_hidden_recog_2, n_z], dtype=tf.float32)}
        all_weights['biases_recog'] = {
            'b1': tf.get_variable("b1_recog", [n_hidden_recog_1], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'b2': tf.get_variable("b2_recog", [n_hidden_recog_2], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'out_mean': tf.get_variable("out_mean_biases_recog", [n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'out_log_sigma': tf.get_variable("out_log_sigma_biases_recog", [n_z], initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        all_weights['weights_gener'] = {
            'h1': tf.get_variable("h1_gener", [n_z, n_hidden_gener_1], dtype=tf.float32),
            'h2': tf.get_variable("h2_gener", [n_hidden_gener_1, n_hidden_gener_2], dtype=tf.float32),
            'out_mean': tf.get_variable("out_mean_weights_gener", [n_hidden_recog_2, n_input], dtype=tf.float32),
            'out_log_sigma': tf.get_variable("out_log_sigma_weights_gener", [n_hidden_recog_2, n_input], dtype=tf.float32)}
        all_weights['biases_gener'] = {
            'b1': tf.get_variable("b1_gener", [n_hidden_gener_1], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'b2': tf.get_variable("b2_gener", [n_hidden_gener_2], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'out_mean': tf.get_variable("out_mean_biases_gener", [n_input], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            'out_log_sigma': tf.get_variable("out_log_sigma_biases_gener", [n_input], initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.matmul(self.x, weights['h1']) + biases['b1'])
        layer_2 = self.transfer_fct(tf.matmul(layer_1, weights['h2'] + biases['b2']))
        z_mean = tf.matmul(layer_2, weights['out_mean']) + biases['out_mean']
        z_log_sigma_sq = tf.matmul(layer_2, weights['out_log_sigma']) + biases['out_log_sigma']
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.matmul(self.z, weights['h1']) + biases['b1'])
        layer_2 = self.transfer_fct(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        x_reconstr_mean = \
            tf.sigmoid(tf.matmul(layer_2, weights['out_mean'] + biases['out_mean']))
        #
        # x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        x_reconstr_log_sigma_sq = tf.matmul(layer_2, weights['out_log_sigma']) + biases['out_log_sigma']
        return x_reconstr_mean, x_reconstr_log_sigma_sq

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # reconstr_loss = \
        #     -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
        #                    + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
        #                    1)
        Q_phi = distributions.MultivariateNormalDiag(self.x, tf.sqrt(tf.exp(self.x_reconstr_logsigma_sq)))
        self.log_prob_reconst = -0.5 * (self.x.get_shape()[1].value * 2 * np.pi + tf.reduce_sum(tf.exp(self.x_reconstr_logsigma_sq), axis=1) +
                                        tf.reduce_sum(tf.square(self.x - self.x_reconstr_mean) / tf.exp(self.x_reconstr_logsigma_sq), axis=1))


        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-9 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-9 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq * 2
                                           - tf.square(self.z_mean)
                                           - tf.square(tf.exp(self.z_log_sigma_sq)), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer


        self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost, prob_reconst = self.sess.run((self.optimizer, self.cost, self.log_prob_reconst),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

def train(network_architecture, learning_rate=0.0001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            pendulum_dataset_batch = pendulum_dataset.next_batch(batch_size)
            batch_x = [np.reshape(before_img, (-1, 3200)) for ind, (before_img,control_signal, after_image, before_states, after_states) in enumerate(pendulum_dataset_batch)]
            batch_x = np.squeeze(np.array(batch_x, np.float32), axis=1) / 255.
            # for i in range(len(batch_x)):
            #     batch_x[i][batch_x[i] != 1] = 0.
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_x)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

network_architecture = \
    dict(n_hidden_recog_1=800, # 1st layer encoder neurons
         n_hidden_recog_2=800, # 2nd layer encoder neurons
         n_hidden_gener_1=800, # 1st layer decoder neurons
         n_hidden_gener_2=800, # 2nd layer decoder neurons
         n_input=3200, # MNIST data input (img shape: 28*28)
         n_z=3)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=100)
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

# plt.figure(figsize=(8, 12))
# for i in range(5):
#
#     plt.subplot(5, 2, 2*i + 1)
#     plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
#     plt.title("Test input")
#     plt.colorbar()
#     plt.subplot(5, 2, 2*i + 2)
#     plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
#     plt.title("Reconstruction")
#     plt.colorbar()
# plt.tight_layout()
# plt.show()
#
#
# x_sample, y_sample = mnist.test.next_batch(5000)
# z_mu = vae.transform(x_sample)
# fig = plt.figure(figsize=(10, 10))
# ax = plt.subplot(111, projection='3d')
# p = ax.scatter(z_mu[:, 0], z_mu[:, 1], z_mu[:, 2], c=np.argmax(y_sample, 1))
# fig.colorbar(p)
# plt.grid()
# plt.show()