import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.distributions as distributions
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from datasets import GymPendulumDatasetV2

plt.style.use('ggplot')

np.random.seed(0)
tf.set_random_seed(0)

# # # Load MNIST data in a format suited for tensorflow.
# # # The script input_data is available under this URL:
# # # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('MNIST_data/', one_hot=True)
x_sample, y_sample = mnist.test.next_batch(2000)
n_samples = mnist.train.num_examples

pendulum_dataset = GymPendulumDatasetV2('data/pendulum_markov_train')
n_samples = len(pendulum_dataset)


# dataset=PlaneData("plane2.npz","env1.png")
# dataset.initialize()
# x_dim,u_dim,T=get_params()
# z_dim=2 # latent space dimensionality

flags = tf.app.flags
flags.DEFINE_string('image_width', pendulum_dataset.width, 'Width for the images.')
flags.DEFINE_string('image_height', pendulum_dataset.height, 'Height for the images.')
flags.DEFINE_string('log_dir' , '/home/bml/yi_zheng/log', 'Directory for log files.')
flags.DEFINE_boolean('dynamics' , False, 'whether to add transition dynamics.')
flags.DEFINE_boolean('deterministic_prediction', True, 'use transition dynamics deterministically.')
flags.DEFINE_boolean('dyanmics_KL_constraint', False, 'add kl diverngence between transition dynamics and encoder net.')
flags.DEFINE_integer('latent_dim', 3,  'Dimensionality for the latent state.')
flags.DEFINE_integer('input_dim', pendulum_dataset.width * pendulum_dataset.height, 'Dimensionality for the input')
flags.DEFINE_integer('control_dim', pendulum_dataset.action_dim, 'Dimensionality for the control signal')
flags.DEFINE_integer('minibatch_size', 100, 'Minibatch size.')
flags.DEFINE_integer('n_samples', n_samples, 'Number of samples per-data point of X.')
flags.DEFINE_integer('print_every', 1000, 'Print every n iterations.')
flags.DEFINE_integer('n_iterations', 1000000, 'number of iterations')
flags.DEFINE_integer('hidden_layers_num', 2, 'number of hidden layers')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate for the optimizer')

FLAGS = flags.FLAGS
network_architecture = \
    dict(n_hidden_recog_1=800, # 1st layer encoder neurons
         n_hidden_recog_2=800, # 2nd layer encoder neurons
         n_hidden_gener_1=800, # 1st layer decoder neurons
         n_hidden_gener_2=800, # 2nd layer decoder neurons
         n_input=FLAGS.input_dim, # MNIST data input (img shape: 28*28)
         n_z=FLAGS.latent_dim)  # dimensionality of latent space




def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def orthogonal_initializer(scale = 1.1):
    """
    reference from Lasagne and Keras, Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, 2013
    """
    def _initializer(shape, dtype=tf.float32):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        print('Warning -- You have opted to use the orthogonal_initializer function')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

class NormalDistribution(object):
    """
    Represent a multivariate Gaussian distribution parameterized by (mu, Cov).
    . If Cov matrix is diagonal, Cov = (sigma).^2. Otherwisr, Cov = A*(sigma).^2*A, where 
    A = (I+v*r^T).
    """
    def __init__(self, mu, sigma, logsigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        dim = mu.get_shape()
        if v is None:
            v = tf.constant(0., shape=dim)
        if r is None:
            r = tf.constant(0., shape=dim)
        self.v = v
        self.r = r


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_function=tf.nn.softplus,
                 learning_rate=FLAGS.learning_rate, minibatch_size=FLAGS.minibatch_size):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_function
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [self.minibatch_size, FLAGS.input_dim])
        self.u = tf.placeholder(tf.float32, [FLAGS.minibatch_size, FLAGS.control_dim])  # control at time step t
        self.x_next = tf.placeholder(tf.float32,[FLAGS.minibatch_size, FLAGS.input_dim])  # state at time step t+1

        self.z_sample = None
        self.x_recons = None
        self.z_predict = None
        self.x_predict = None
        self.W_z = None

        # Create autoencoder network
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self._create_encoder_net(self.network_weights)
        if FLAGS.dynamics == True:
            self._create_transition_dynamics()
        self._create_decoder_net(self.network_weights)
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        # Create autoencoder network

        # Define the variational lower-bound
        self._create_vlb()
        # Define the optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {'h1': tf.get_variable("h1_recog", [n_input, n_hidden_recog_1], dtype=tf.float32),
                                        'h2': tf.get_variable("h2_recog", [n_hidden_recog_1, n_hidden_recog_2], dtype=tf.float32),
                                        'out_mean': tf.get_variable("out_mean_weights_recog", [n_hidden_recog_2, n_z], dtype=tf.float32),
                                        'out_log_sigma': tf.get_variable("out_log_sigma_weights_recog", [n_hidden_recog_2, n_z], dtype=tf.float32)}
        all_weights['biases_recog'] = {'b1': tf.get_variable("b1_recog", [n_hidden_recog_1], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                                       'b2': tf.get_variable("b2_recog", [n_hidden_recog_2], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                                        'out_mean': tf.get_variable("out_mean_biases_recog", [n_z], initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32),
                                        'out_log_sigma': tf.get_variable("out_log_sigma_biases_recog", [n_z],
                                        initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        all_weights['weights_gener'] = {'h1': tf.get_variable("h1_gener", [n_z, n_hidden_gener_1], dtype=tf.float32),
                                        'h2': tf.get_variable("h2_gener", [n_hidden_gener_1, n_hidden_gener_2], dtype=tf.float32),
                                        'out_mean': tf.get_variable("out_mean_weights_gener", [n_hidden_recog_2, n_input], dtype=tf.float32),
                                        'out_log_sigma': tf.get_variable("out_log_sigma_weights_gener", [n_hidden_recog_2, n_input],
                                        dtype=tf.float32)}
        all_weights['biases_gener'] = {'b1': tf.get_variable("b1_gener", [n_hidden_gener_1], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                                       'b2': tf.get_variable("b2_gener", [n_hidden_gener_2], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                                        'out_mean': tf.get_variable("out_mean_biases_gener", [n_input], initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32),
            'out_log_sigma': tf.get_variable("out_log_sigma_biases_gener", [n_input],
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        return all_weights

    def _create_encoder_net(self, network_weights):
        # Use the encoder net to determine the mean and log variance
        # of Gaussian distribution in the latent space
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights["weights_recog"],
                                                                        network_weights["biases_recog"], self.x)
        # Draw z samples from Gaussian distribution
        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal(self.z_mean.get_shape(), mean=0, stddev=1, dtype=tf.float32)
        self.z_sample = self.z_mean + tf.exp(self.z_log_sigma_sq) * eps

    def _create_decoder_net(self, network_weights):
        # Use the encoder net to determine the mean and log variance
        # of Gaussian distribution in the latent space
        self.x_reconstr_mean, self.x_reconstr_logsigma_sq = self._generator_network(network_weights["weights_gener"],
                                                                        network_weights["biases_gener"], self.z_sample)

    def _recognition_network(self, weights, biases, x, share=None):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        with tf.variable_scope("encoder_net", reuse=share):
            layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']),
                                               biases['b1']))
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                               biases['b2']))
            z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                            biases['out_mean'])
            z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                                    biases['out_log_sigma'])
            return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases, z_sample, type='Gaussian', share=None):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        with tf.variable_scope("decoder_net", reuse=share):
            layer_1 = self.transfer_fct(tf.add(tf.matmul(z_sample, weights['h1']),
                                        biases['b1']))
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                        biases['b2']))
            x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                            biases['out_mean']))
            if type == "Gaussian":
                x_reconstr_logsigma_sq = tf.matmul(layer_2, weights['out_log_sigma']) + biases['out_log_sigma']
                print ("decoder net outputs a Gaussian distribution.")
                return (x_reconstr_mean, x_reconstr_logsigma_sq)
            else:
                print ("decoder net outputs a Bernoulli distribution.")
                return x_reconstr_mean

    def _create_transition_dynamics(self, share=None):
        with tf.variable_scope("transition_dynamics", reuse=share):
            self.W_z = tf.get_variable('W_z', [FLAGS.latent_dim, FLAGS.latent_dim], initializer=tf.constant_initializer(0.0))
            self.W_u = tf.get_variable('W_u', [FLAGS.control_dim, FLAGS.latent_dim], initializer=tf.constant_initializer(0.0))
            self.b = tf.get_variable('b', [FLAGS.latent_dim], initializer=tf.constant_initializer(0.0))
            self.z_predict = tf.matmul(self.z_sample, self.W_z) + tf.matmul(self.u, self.W_u) + self.b
            return self.z_predict

    def _create_vlb(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
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
        latent_loss = -0.5 * tf.reduce_sum(1 + 2 * self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.square(tf.exp(self.z_log_sigma_sq)), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch

    def _create_loss_optimizer(self):
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
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
                             feed_dict={self.z_sample: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})



def train(nwetwork_architeture, learning_rate=FLAGS.learning_rate, minibatch_size=FLAGS.minibatch_size,
          training_epochs=100, display_step = 5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 minibatch_size=minibatch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / minibatch_size)
        # Loop over all batches
        for i in range(total_batch):
            pendulum_dataset_batch = pendulum_dataset.next_batch(minibatch_size)
            batch_x = [np.reshape(before_img, (-1, 3200)) for
                       ind, (before_img, control_signal, after_image, before_states, after_states) in
                       enumerate(pendulum_dataset_batch)]
            batch_x = np.squeeze(np.array(batch_x, np.float32), axis=1) / 255.
            # for i in range(len(batch_x)):
            #     batch_x[i][batch_x[i] != 1] = 0.
            batch_xs, _ = mnist.train.next_batch(minibatch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_x)
            # Compute average loss
            avg_cost += cost / n_samples * minibatch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

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
# x_sample, y_sample = mnist.test.next_batch(2000)
# z_mu = vae.transform(x_sample)
# fig = plt.figure(figsize=(10, 10))
# ax = plt.subplot(111, projection='3d')
# p = ax.scatter(z_mu[:, 0], z_mu[:, 1], z_mu[:, 2], c=np.argmax(y_sample, 1))
# fig.colorbar(p)
# plt.grid()
# plt.show()
