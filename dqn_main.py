#!/usr/bin/python3.5
import math

import numpy as np

from dqn_env import DqnEnv
from dqn_agent import Agent
import os


##### DQN #####

# ~~~ Embroidery specific ~~~ #
from pytsp.tsp_computer import TSPComputer

test = False
debug = False
two_by_two = False
# 1st layer
layer_1 = np.load(os.path.join('data', '7_17.npy'))
if two_by_two:
    layer_1 = np.array([[0, 255], [255, 255]])

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

agent = Agent(action_space, gamma, minibatch_size, observation_shape, replay_memory_capacity, test)

initial_epsilon = 1 if not test else 0
final_epsilon = 0.1 if not test else initial_epsilon
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

nb_episodes = int(300e3) if not test else 1
episodes_per_epoch = int(1e4)
if two_by_two:
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

    # print(o_t)

    dqn_env = DqnEnv()
    ep_reward = 0.0
    while not episode_done and not (test and t >= 1000):
        # Choose an action based on observation and exploration probability
        a_t = agent.act(o_t)

        a_t_mapped = actions[a_t]

        # print(a_t_mapped)

        # Take a step in the environment based on chosen action and observe the outcome
        o_tp1, r_tp1, episode_done = dqn_env.step(o_t, a_t_mapped)

        # print(o_tp1)
        # print(r_tp1)
        # print("\n")

        # Update the replay memory

        o_t_reshaped = np.reshape(o_t, (x_dim, y_dim, 2))
        o_tp1_reshaped = np.reshape(o_tp1, (x_dim, y_dim, 2))

        agent.update_replay_memory(o_t_reshaped, a_t, r_tp1, o_tp1_reshaped, episode_done)

        ep_reward += r_tp1

        if episode_done:
            if not test and max_reward <= ep_reward:
                str_out = 'max_reward={} <= ep_reward={}'.format(max_reward, ep_reward)
                max_reward = ep_reward
                agent.save(global_step=t)
                if debug:
                    print('Saved graph: ')
                    agent.print_vars()
                    print('saving done')
                    print('steps: ', dqn_env.steps, ', reward: ' + str(ep_reward), 'str_out=', str_out)
            completed_episodes += 1
            episode_reward.append(ep_reward)

            episode_length.append(t)

        # No changes to either Q and target Q network until the replay memory has been filled
        # with <replay_start_size> random transitions
        if not test and number_of_observations_experienced >= replay_start_size:

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

    if completed_episodes % episodes_per_epoch == 0 or test:
        print('Episode ', completed_episodes, ', mean reward over last ', episodes_per_epoch, ' episodes: ',
              np.mean(episode_reward[-episodes_per_epoch:]) if len(episode_reward) >= episodes_per_epoch else 0)
        print('Epsilon: ', agent.epsilon)
        print('RL steps: ', dqn_env.steps, ', reward: ' + str(ep_reward), ', done: ', episode_done)
        print('steps', len(dqn_env.steps) , 'coords: ', len(tsp_computer.coords.keys()))
        if episode_done and len(dqn_env.steps) == len(tsp_computer.coords.keys()):
            print('tsp_cost', tsp_computer.tsp_cost(dqn_env.steps[0]))
            print('rl_cost', tsp_computer.rl_cost(dqn_env.steps))


save_dir = 'out'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'dqn2_toy_reward'), np.array(episode_reward))
np.save(os.path.join(save_dir, 'dqn2_toy_length'), np.array(episode_length))