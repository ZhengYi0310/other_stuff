import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Dropout
from keras import initializers
from keras import optimizers

def run_export_policy_gather_data(config):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(config['expert_policy_file'])
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(config['env'])
        max_steps = config['max_timesteps'] or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        steps_list = []
        for i in tqdm(range(config['num_rollouts'])):
            print('rollout: ', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if config['do_rendering']:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            steps_list.append(steps)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns),
                       'steps': np.array(steps_list)}
    pickle.dump(expert_data, open(config['expert_data_path'], 'wb'))
    return expert_data

def train_NN_policy(config, data, env):

    expert_data = pickle.load(open(config['expoert_data_path']))

    # Construct the models
    model = Sequential()
    model.add(Lambda(lambda x: (x - np.mean(x)) / np.std(x)))
    model.add(Dense(units=64, input_dim=env.observation_space.shape[0],
                    kernel_initializer=initializers.he_normal(seed=None)))
    model.add(Activation('relu'))
    model.add(Dense(units=64, input_dim=64,
                    kernel_initializer=initializers.he_normal(seed=None)))
    model.add(Activation('relu'))
    model.add(Dropout(config['drop_out_rate']))
    model.add(Dense(units=env.action_space.shape[0]))

    opt = optimizers.Adagrad(lr=config['learning_rate'], epsilon=config['epsilon'], decay=config['decay'])
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', 'accuracy'])


