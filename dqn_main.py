#!/usr/bin/python3.5
import math

import numpy as np

from dqn_env import DqnEnv
from dqn_agent import Agent
import os
import copy
from utils import logging_setup
import logging


##### DQN #####

# ~~~ Embroidery specific ~~~ #
from pytsp.tsp_computer import TSPComputer

log = logging.getLogger(__name__)

logging_setup.init()


class Config:
    def __init__(self, data_file, step_reward, fast_fail, reuse_weights, test, debug):
        self.data_file = data_file
        self.step_reward = step_reward
        self.fast_fail = fast_fail
        self.reuse_weights = reuse_weights
        self.test = test
        self.debug = debug

    def get_name(self):
        return 'data_file={},step_reward={},fast_fail={},reuse_weights={},test={}'.\
            format(self.data_file, self.step_reward, self.fast_fail,
                   self.reuse_weights is not False, self.test)

    def get_out_dir(self):
        return os.path.join('out', self.get_name())

    def get_checkpoints_save_dir(self):
        return os.path.join('checkpoints', self.get_name())

    def get_checkpoints_load_dir(self):
        c2 = copy.copy(self)
        if self.test:
            c2.test = False
            return os.path.join('checkpoints', c2.get_name())
        elif self.reuse_weights:
            return os.path.join('checkpoints', c2.reuse_weights)

    def load_file(self):
        if self.data_file == '2x2':
            return np.array([[0, 255], [255, 255]]) # tree cells for LKH
        else:
            a = np.load(os.path.join('data', self.data_file))
            return a


c = Config(
    data_file='0_13.npy',
    step_reward=-0.1, #-0.1, -0.5, -1
    fast_fail=True,
    reuse_weights=False, # False or folder name
    test=True,
    debug=False
)

log.debug('>>>> RUNNING {}<<<<'.format(c.get_name()))

# 1st layer
layer_1 = c.load_file()

tsp_computer = TSPComputer(layer_1)

x_dim, y_dim = layer_1.shape

# 2nd layer

layer_2 = np.zeros_like(layer_1)

original_position = (0, 0)

layer_2[original_position] = 255

# Base observation

base_observation = np.zeros((2, x_dim, y_dim))

base_observation[0] = layer_1
base_observation[1] = layer_2

actions = {0: 'u', 1: 'd', 2: 'r', 3: 'l', 4: 's'}

# ~~~ ~~~ #

# ~~~ Initialize DQN agent ~~~ #

action_space = len(actions)
gamma = .99
minibatch_size = 32
observation_shape = (base_observation.shape[1], base_observation.shape[2], base_observation.shape[0])
replay_memory_capacity = int(1e6)

agent = Agent(action_space, gamma, minibatch_size, observation_shape, replay_memory_capacity, c)

initial_epsilon = 1 if not c.test else 0
final_epsilon = 0.1 if not c.test else initial_epsilon
final_exploration_frame = int(1e6)
delta_epsilon = (initial_epsilon - final_epsilon) / final_exploration_frame

# Epsilon greedy exploration
agent.set_exploration_scheme(initial_epsilon, final_epsilon, delta_epsilon)

# Initialize neural network weights and biases
agent.initialize_neural_net_weights()

# ~~~ ~~~ #

# ~~~ Initialize parameters, variables, environment ~~~ #

# 30 episodes
replay_start_size = int(50e3)

# 5 episodes
target_network_update_frequency = int(10e3)

update_frequency = 4

nb_episodes = int(300e3) if not c.test else 1
episodes_per_epoch = int(1e4)
if layer_1.shape == (2, 2):
    div_by = 1
    nb_episodes = int(nb_episodes / div_by)
    episodes_per_epoch = int(episodes_per_epoch / div_by)

completed_episodes = 0

episode_reward = []
episode_length = []

number_of_observations_experienced = 0

# ~~~ ~~~ #

max_reward = -math.inf

while completed_episodes < nb_episodes:

    t = 0
    o_t = np.copy(base_observation)
    episode_done = False

    # log.debug((o_t)

    dqn_env = DqnEnv(c)
    ep_reward = 0.0
    while not episode_done and not (c.test and t >= 1000):
        # Choose an action based on observation and exploration probability
        a_t = agent.act(o_t)

        a_t_mapped = actions[a_t]

        # log.debug((a_t_mapped)

        # Take a step in the environment based on chosen action and observe the outcome
        o_tp1, r_tp1, episode_done = dqn_env.step(o_t, a_t_mapped)

        # log.debug((o_tp1)
        # log.debug((r_tp1)
        # log.debug(("\n")

        # Update the replay memory

        o_t_reshaped = np.reshape(o_t, (x_dim, y_dim, 2))
        o_tp1_reshaped = np.reshape(o_tp1, (x_dim, y_dim, 2))

        agent.update_replay_memory(o_t_reshaped, a_t, r_tp1, o_tp1_reshaped, episode_done)

        ep_reward += r_tp1

        if episode_done:
            if not c.test and max_reward <= ep_reward:
                str_out = 'max_reward={} <= ep_reward={}'.format(max_reward, ep_reward)
                max_reward = ep_reward
                agent.save(global_step=t)
                if c.debug:
                    log.debug('Saved graph: ')
                    agent.print_vars()
                    log.debug('saving done')
                    log.debug('steps: {}, reward: {}, str_out={}'.format(dqn_env.steps, ep_reward, str_out))
            completed_episodes += 1
            episode_reward.append(ep_reward)

            episode_length.append(t)

        # No changes to either Q and target Q network until the replay memory has been filled
        # with <replay_start_size> random transitions
        if not c.test and number_of_observations_experienced >= replay_start_size:

            # Train the Q network once every <update_frequency> iterations
            if number_of_observations_experienced % update_frequency == 0:
                agent.train()

            # Update target Q network weights once every <target_network_update_frequency> iterations to be equal
            # to the Q network weights
            if number_of_observations_experienced % target_network_update_frequency == 0:
                agent.update_target_Q_network_weights()

            # Decrease exploration probability
            agent.update_exploration_value()

        # New timestep
        t += 1
        number_of_observations_experienced += 1
        o_t = o_tp1

    if completed_episodes % episodes_per_epoch == 0 or c.test:
        log.debug('Episode {}, mean reward over last {} episodes: {}'.format(completed_episodes, episodes_per_epoch,
              np.mean(episode_reward[-episodes_per_epoch:]) if len(episode_reward) >= episodes_per_epoch else 0))
        log.debug('Epsilon: {}'.format(agent.epsilon))
        log.debug('RL steps: {}, reward: {}, done: {}'.format(dqn_env.steps, ep_reward, episode_done))
        log.debug('Steps: {}, coords: {}'.format(len(dqn_env.steps), len(tsp_computer.coords.keys())))
        if episode_done and len(dqn_env.steps) == len(tsp_computer.coords.keys()):
            log.debug('tsp_cost {}'.format(tsp_computer.tsp_cost(dqn_env.steps[0])))
            print('rl_cost {}'.format(tsp_computer.rl_cost(dqn_env.steps)))

    save_dir = c.get_out_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'episode_reward'), np.array(episode_reward))
    np.save(os.path.join(save_dir, 'episode_length'), np.array(episode_length))