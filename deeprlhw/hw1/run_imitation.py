import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import tf_util
import gym
import load_policy
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Dropout
from keras import initializers
from keras import optimizers
from keras import callbacks


def run_export_policy_gather_data(config):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(config['expert_policy_file'])
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

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
                if config['do_rendering_expert']:
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

    expert_data = pickle.load(open(config['expert_data_path']), 'rb')

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

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2)

    model.fit(expert_data['observations'], expert_data['actions'].reshape(-1, env.action_space.shape[0]), validation_split=0.2,
              batch_size=40, epochs=100, verbose=2, shuffle=True, callbacks=[early_stopping])

    model.save('behavior_cloning_model.h5')

    return model

def test_NN_policy(model, config, env):
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
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if config['do_rendering_imitation']:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        steps_list.append(steps)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    policy_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns': np.array(returns),
                   'steps': np.array(steps_list)}


    pickle.dump(policy_data, open(config['policy_data_path'], 'wb'))
    return policy_data

def compute_one_data_set_stats(data):
    mean = data['returns'].mean()
    std = data['returns'].std()
    step = data['steps']
    avg_percentage_fullsteps = (step / (step.max())).mean()

    return pd.Series({'mean_returns': mean, 'std_returns': std, 'avg_percentage_fullsteps': avg_percentage_fullsteps})

def analyze_one_experiment_data(config):
    imitation_stats = compute_one_data_set_stats(pickle.load(open(config['policy_data_path']), 'rb'))
    expert_stats = compute_one_data_set_stats(pickle.load(open(config['expert_data_path']), 'rb'))

    data_frame = pd.DataFrame({'expert_stats': expert_stats, 'imitation_stats': imitation_stats})

    print('Analyzing stats of a single experiment for {}'.format(config['env']))
    print data_frame

def load_default_config(env):
    return {
        'env_name': env,
        'expert_policy_file': 'experts/{}.pkl'.format(env),
        'envname': env,
        'do_rendering_expert': True,
        'do_rendering_policy': False,
        'num_rollouts': 30,
        'use_cached_data_for_training': True,
        'cached_data_path': 'data/{}-cached.p'.format(env),
        'expert_data_path': 'data/{}-expert.p'.format(env),
        'imitation_data_path': 'data/{}-imitation.p'.format(env),
        # neural net params
        'learning_rate': 0.001,
        'epochs': 30,
        'drop_out_rate': 0.2
    }

def get_learning_rate_config_grid(env):
    learning_rate_grid = np.logspace(-5, 0, 20)
    configs = []
    for lr in learning_rate_grid:
        config = load_default_config(env)
        config['use_cached_data_for_training'] = True
        config['imitation_data_path'] = 'data/{}-imitation-lr={:.8f}.p'.format(env, lr)
        config['learning_rate'] = lr
        configs.append(config)
    return configs

def get_training_epoch_config_grid(env):
    training_epoch_grid = [1, 2, 5, 10, 15 ,20 ,30, 50, 80, 100]
    configs = []
    for epochs in training_epoch_grid:
        config = load_default_config(env)
        config['use_cached_data_for_training'] = True
        config['imitation_data_path'] = 'data/{}-imitation-epochs={:.8f}.p'.format(env, epochs)
        config['epochs'] = epochs
        configs.append(config)
    return configs

def get_dropout_rates_config_grid(env):
    dropout_grid = [0.1, 0.15, 0.3, 0.5, 0.7, 0.9, 0.95]
    configs = []
    for dropout in dropout_grid:
        config = load_default_config(env)
        config['use_cached_data_for_training'] = True
        config['imitation_data_path'] = 'data/{}-imitation-dropout={:.8f}.p'.format(env, dropout)
        config['drop_out_rate'] = dropout
        configs.append(config)
    return configs





