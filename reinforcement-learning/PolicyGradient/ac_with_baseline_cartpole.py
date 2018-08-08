# code adapted from:
# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

env = CliffWalkingEnv()
env = gym.make('CartPole-v0')

# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)  # reproducible


number_features = env.observation_space.shape[0]
number_actions = env.action_space.n
gamma = 0.9     # reward discount in TD error
learning_rate = 0.01


# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 999

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            #self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            #self.loss = -tf.log(self.picked_action_prob) * self.target

            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.target = tf.placeholder(tf.float32, [1, 1], "target")

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.loss = tf.squared_difference(self.v, self.target)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.v, {self.state: state})

    def learn(self, state, target):
        state = state[np.newaxis, :]
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss



def actor_critic(env, actor, critic, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        actor: Policy Function to be optimized
        critic: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()

        episode = []
        track_r = []

        # One step in the environment
        for t in range(max_ep):

            # Take a step
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            track_r.append(reward)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Calculate TD Target
            value_next = critic.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - critic.predict(state)

            # Update the value estimator
            critic.learn(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            actor.learn(state, action, td_error)

            # Print out which step we're on, useful for debugging.
            #print("\rStep {} @ Episode {}/{} ({})".format(
            #    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
            #env.render()

            if done:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))

                break

            state = next_state

    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)

with tf.Session() as sess:
    actor = Actor(sess, number_features, number_actions)
    critic = Critic(sess, number_features)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, actor, critic, total_episodes)