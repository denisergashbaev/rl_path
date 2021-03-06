# Code is based on
# https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb
# https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os
from gym import wrappers
env = gym.make('CartPole-v0')
# force=True to overwrite the previous videos
env = wrappers.Monitor(env, 'gym-videos', force=True)

# Policy gradient has high variance, seed for reproducability
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)  # reproducible
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    # r represents all the rewards collected throughout the episode
    # np.zeros_like -- Return an array of zeros with the same shape and type as a given array.
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent:
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        # this is not used in reinforce
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training procedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()  # Clear the Tensorflow graph.

#https://github.com/openai/gym/wiki/CartPole-v0
# 4 states: cart position, cart velocity, pole angle, pole velocity at tip
# 2 actions: 0 - push cart to the left, 1 - push cart to the right
myAgent = agent(lr=1e-2, s_size=env.observation_space.shape[0], a_size=env.action_space.n, h_size=8)  # Load the agent.

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Probabilistically pick an action given our network outputs.
            # a_dist = [[0.49502707 0.504973  ]]
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            # a = 0.49502707
            a = np.random.choice(a_dist[0], p=a_dist[0])
            # a = 0
            a = np.argmax(a_dist == a)

            s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
            #env.render()
            # print('done={}'.format(d))
            idx_s, idx_a, idx_r = 0, 1, 2
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            # https://github.com/openai/gym/wiki/CartPole-v0
            # episode termination:
            # 1) Pole Angle is more than ±12°
            # 2) Cart Position is more than ±2.4(center of the cart reaches the edge of the display)
            # 3) Episode length is greater than 200
            if d:
                # Update the network.
                ep_history = np.array(ep_history)
                # take the column of rewards from the ep_history
                # and apply discounting to it
                # todo: i think it's monte carlo so i need to go over it
                ep_history[:, idx_r] = discount_rewards(ep_history[:, idx_r])
                feed_dict = {myAgent.reward_holder: ep_history[:, idx_r],
                             myAgent.action_holder: ep_history[:, idx_a], myAgent.state_in: np.vstack(ep_history[:, idx_s])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_lenght.append(j)
                break

        # Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
        np.save(os.path.join('.', 'reinforce_total_reward2'), np.array(total_reward))