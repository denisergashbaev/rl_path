import os
import tensorflow as tf
import numpy as np
import random
import shutil
import logging

log = logging.getLogger(__name__)



# ~~~~~ Deep Q-Learning (DQN) Agent ~~~~~ #

class Agent:
    def __init__(self, action_space, gamma, minibatch_size, observation_shape, replay_memory_capacity, c):
        self.c = c
        # Agent variables
        self.action_space = action_space
        self.replay_memory_capacity = replay_memory_capacity
        self.minibatch_size = minibatch_size
        self.observation_shape = observation_shape

        # Initialize replay memory
        self.replay_memory = ReplayBuffer(replay_memory_capacity)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        from tensorflow.python.client import device_lib
        log.debug('device: {}'.format(device_lib.list_local_devices()))
        # ~~~ Create placeholders  ~~~ #

        with tf.device('/GPU:0'):
            a_test = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b_test = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c_test = tf.matmul(a_test, b_test)
        # Runs the op.
        log.debug('device_placement: {}'.format(self.sess.run(c_test)))

        self.o_t_ph = tf.placeholder(tf.uint8, [None] + list(observation_shape))
        self.a_t_ph = tf.placeholder(tf.int32, [None])
        self.r_tp1_ph = tf.placeholder(tf.float32, [None])
        self.o_tp1_ph = tf.placeholder(tf.uint8, [None] + list(observation_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # Cast to float to cut the runtime by almost half
        self.o_t_ph_float = tf.cast(self.o_t_ph, tf.float32) / 255.0
        self.o_tp1_ph_float = tf.cast(self.o_tp1_ph, tf.float32) / 255.0

        # ~~~ ~~~ #

        # ~~~ Build computation graph ~~~ #

        # Q(o_t, *, theta)
        self.Q_o_t = self.Q(self.o_t_ph_float, scope='q')

        # Q(o_t+1, *, theta_bar)
        self.target_Q_o_tp1 = self.Q(self.o_tp1_ph_float, scope='target_q')

        # Create weight collections to separate the Q network and target Q network
        self.Q_network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
        self.target_Q_network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q')

        # Q(o_t, a_t, theta)
        self.Q_o_t_a_t = tf.reduce_sum(self.Q_o_t * tf.one_hot(self.a_t_ph, action_space), axis=1)

        # Q(o_t+1, a_t+1, theta_bar) = max_a Q(o_t+1, a, theta_bar)
        self.target_Q_o_tp1_a_tp1 = tf.reduce_max(self.target_Q_o_tp1, axis=1)

        # J = [ r_tp1 + gamma * Q(o_t+1, a_t+1, theta_bar) - Q(o_t, a_t, theta) ] ^ 2
        self.loss_op = tf.losses.mean_squared_error(
            self.r_tp1_ph + gamma * self.target_Q_o_tp1_a_tp1 * (1 - self.done_mask_ph), self.Q_o_t_a_t)

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # Compute gradients - grad_theta J
        self.gradients = self.optimizer.compute_gradients(loss=self.loss_op, var_list=self.Q_network_weights)

        # Gradient clipping
        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
                self.gradients[i] = (tf.clip_by_norm(grad, clip_norm=10), var)

        # Apply gradients
        self.train_op = self.optimizer.apply_gradients(self.gradients)

        self.saver = tf.train.Saver()
        if not c.test:
            self.save_dir = self.c.get_checkpoints_save_dir()
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                shutil.rmtree(self.save_dir)

    # Q Function approximator based on some observation and network weights (scope)
    def Q(self, observation, scope):

        # Build neural net under a scope to separate both Q and target Q networks
        with tf.variable_scope(scope):

            if observation.shape.dims[1:3] == [2, 2]:
                # 2x2 images network
                # m x 2x2x2 -> m x 2x2x8
                conv1 = tf.contrib.layers.conv2d(inputs=observation, num_outputs=8, kernel_size=(1, 1), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 2x2x8 -> m x 2x2x16
                conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=16, kernel_size=(1, 1), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 2x2x16 -> m x 1x1x16
                conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=16, kernel_size=(2, 2), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 6x6x16 -> m x 64
                flat1 = tf.contrib.layers.flatten(inputs=conv3)

                # m x 64 -> m x 16
                fc1 = tf.contrib.layers.fully_connected(inputs=flat1, num_outputs=16, activation_fn=tf.nn.relu)

                # m x 16 -> m x action_space
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.action_space, activation_fn=None)

            else:
                # 10x10 images network
                # m x 10x10x2 -> m x 10x10x8
                conv1 = tf.contrib.layers.conv2d(inputs=observation, num_outputs=8, kernel_size=(1, 1), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 10x10x8 -> m x 8x8x12
                conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=12, kernel_size=(3, 3), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 8x8x12 -> m x 6x6x16
                conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=16, kernel_size=(3, 3), stride=(1, 1),
                                                 padding='VALID', activation_fn=tf.nn.relu)

                # m x 6x6x16 -> m x 576
                flat1 = tf.contrib.layers.flatten(inputs=conv3)

                # m x 576 -> m x 128
                fc1 = tf.contrib.layers.fully_connected(inputs=flat1, num_outputs=128, activation_fn=tf.nn.relu)

                # m x 128 -> m x action_space
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.action_space, activation_fn=None)

            Q_obs = fc2
            return Q_obs

    # Initialize Q network weights and target Q network weights
    def initialize_neural_net_weights(self):
        # Initialize neural network weights and biases ...
        self.sess.run(tf.global_variables_initializer())
        # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model/33763208#33763208
        folder_name=None
        if self.c.reuse_weights:
            folder_name = self.c.get_checkpoints_load_dir()
        if self.c.test:
            folder_name = self.c.reuse_weights
        if folder_name:
            log.debug('loading checkpoint from {}'.format(folder_name))
            self.saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(os.path.join('checkpoints', folder_name)))

        # ... and make the Q and target Q networks equal
        self.update_target_Q_network_weights(ignore=True)
        self.print_vars()

    def print_vars(self):
        graph = tf.get_default_graph()
        for v in [v for v in tf.trainable_variables()]:
            #print(v[0])
            # It will give tensor object
            var = graph.get_tensor_by_name(v.name)

            # To get the value (numpy array)
            var_value = self.sess.run(var)
            log.debug('var_name: {}, var_value {}: '.format(v.name, var_value))

    # Update target Q network weights to be equal to the Q network weights
    def update_target_Q_network_weights(self, ignore=False):
        if self.c.test and not ignore:
            raise RuntimeError('Cannot update targer q-network weights on testing')

        new_target_Q_network_weights = []

        for Q_w, target_Q_w in zip(sorted(self.Q_network_weights, key=lambda v: v.name),
                                   sorted(self.target_Q_network_weights, key=lambda v: v.name)):
            new_target_Q_network_weights.append(target_Q_w.assign(Q_w))

        copy_op = tf.group(*new_target_Q_network_weights)
        self.sess.run(copy_op)

    # Epsilon greedy exploration
    def set_exploration_scheme(self, initial_epsilon, final_epsilon, delta_epsilon):

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.delta_epsilon = delta_epsilon

        self.epsilon = initial_epsilon

    # Decrease exploration probability
    def update_exploration_value(self):

        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.delta_epsilon

    # Choose an action based on observation and exploration probability
    def act(self, o_t):

        # Explore if random probability is less than the epsilon value ...
        if np.random.rand() <= self.epsilon:
            # Random action from the action space
            a_t = np.random.randint(self.action_space)

        # ... Exploit otherwise
        else:
            # Reshape for Q function approximator
            o_t = np.reshape(o_t, (1,) + self.observation_shape)

            Q_o_t = self.sess.run(self.Q_o_t, feed_dict={self.o_t_ph: o_t})
            a_t = np.argmax(Q_o_t)

        return a_t

        # Update the replay memory

    def update_replay_memory(self, o_t, a_t, r_tp1, o_tp1, done):
        if not self.c.test:
            self.replay_memory.add(o_t, a_t, r_tp1, o_tp1, float(done))

    # Train the Q network once every <update_frequency> iterations
    def train(self):
        if self.c.test:
            raise RuntimeError('Cannot train on testing')

        # Fetch <minibatch_size> samples
        o_t_batch, a_t_batch, r_tp1_batch, o_tp1_batch, done_mask_batch = self.replay_memory.sample(self.minibatch_size)

        # Update Q network
        self.sess.run(self.train_op, feed_dict={self.o_t_ph: o_t_batch,
                                           self.a_t_ph: a_t_batch,
                                           self.r_tp1_ph: r_tp1_batch,
                                           self.o_tp1_ph: o_tp1_batch,
                                           self.done_mask_ph: done_mask_batch})

    def save(self, global_step):
        if self.c.test:
            raise RuntimeError('cannot save network during test')
        # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/04_Save_Restore.ipynb
        save_path = os.path.join(self.save_dir, 'best_reward')
        self.saver.save(sess=self.sess, save_path=save_path, global_step=global_step)


# ~~~~~ Replay Buffer ~~~~~ #

# Software written by OpenAI
# https://github.com/openai/baselines/blob/5d62b5bdaa05be5a834a6c38ec61feb6110f05cc/baselines/deepq/replay_buffer.py


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
