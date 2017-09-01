import numpy as np, tensorflow as tf, tensorflow.contrib.slim as slim, tensorflow.contrib.layers as layer
from functions import *

class Agent:
    def __init__(self, sess, name, input_shape, valid_actions, trainer = None):
        self.sess = sess
        self.name = name
        self.input_shape = input_shape
        self.valid_actions = valid_actions
        self.trainer = trainer
        self.build_model()

    def build_model(self):
        epsilon = 1e-7
        self.states = tf.placeholder(tf.float32, shape=(None, *self.input_shape))

        with tf.variable_scope(self.name):
            # Convolution layers
            '''net = slim.conv2d(inputs=self.states, activation_fn=tf.nn.elu, num_outputs=32, kernel_size=(3, 3), stride=(2, 2), data_format='NCHW')
            net = slim.conv2d(inputs=net, activation_fn=tf.nn.elu, num_outputs=32, kernel_size=(3, 3), stride=(2, 2), data_format='NCHW')
            net = slim.conv2d(inputs=net, activation_fn=tf.nn.elu, num_outputs=32, kernel_size=(3, 3), stride=(2, 2), data_format='NCHW')
            net = slim.conv2d(inputs=net, activation_fn=tf.nn.elu, num_outputs=32, kernel_size=(3, 3), stride=(2, 2), data_format='NCHW')'''
            net = slim.conv2d(inputs=self.states, activation_fn=tf.nn.relu, num_outputs=16, kernel_size=(8, 8), stride=(4, 4), data_format='NCHW')
            net = slim.conv2d(inputs=net, activation_fn=tf.nn.relu, num_outputs=32, kernel_size=(4, 4), stride=(2, 2), data_format='NCHW')

            net = slim.flatten(net)

            # Fully connected layer
            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)

            # Policy network
            self.policy_network = slim.fully_connected(net, len(self.valid_actions), activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)

            # Value network
            self.value_network = tf.squeeze( slim.fully_connected(net, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1), biases_initializer=None) )

        if self.name == 'global':
            self.actions = tf.placeholder(tf.uint8, shape=(None))
            action_onehots = tf.one_hot(self.actions, depth=len(self.valid_actions))
            self.rewards = tf.placeholder(tf.float32, shape=(None))
            self.advantages = tf.placeholder(tf.float32, shape=(None))

            # Action loss
            single_action_prob = tf.reduce_sum(self.policy_network * action_onehots, axis=1)
            log_action_prob = - tf.log(single_action_prob + epsilon) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

            # Value loss
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.value_network))

            # Entropy
            entropy = - tf.reduce_sum(self.policy_network * tf.log(self.policy_network + epsilon), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

            # Total loss (value loss + policy loss)
            self.total_loss = (0.5 * value_loss) + (action_loss - 0.01 * entropy_sum)

            # Train network
            gradients = self.trainer.compute_gradients(loss=self.total_loss)
            gradients = [(tf.clip_by_norm(grad, 50), var) for grad, var in gradients]
            self.train_network = self.trainer.apply_gradients(gradients, global_step=tf.contrib.framework.get_or_create_global_step())

        if self.name != 'global':
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            self.update_local_ops = [local_var.assign(global_var) for global_var, local_var in zip(global_vars, local_vars)]

    def update_from_global(self):
        # Copy variables from global agent to local agent
        self.sess.run(self.update_local_ops)

    def format_state(self, state):
        return np.reshape(state, (-1, *self.input_shape))

    def get_action_value(self, state):
        state = self.format_state(state)
        policy, value = self.sess.run([self.policy_network, self.value_network], {
            self.states: state
        })
        policy = policy[0]

        a = np.random.choice(policy, p=policy)
        actionIndex = np.argmax(policy == a)
        action = self.valid_actions[actionIndex]

        return action, value

    def train(self, states, actions, rewards, advantages):
        self.sess.run([self.train_network], feed_dict={
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        })
