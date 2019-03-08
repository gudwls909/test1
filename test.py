import random
import numpy as np
import tensorflow as tf


class AutoEncoder(object):
	def __init__(self, input_size, output_size, SNR):
		self.input_size = input_size
		self.output_size = output_size
		self.SNR = SNR

		self.states = tf.placeholder(tf.float32, [None, self.state_size])
		self.actions = tf.placeholder(tf.int64, [None])
		self.target = tf.placeholder(tf.float32, [None])

		self.block = self.build_encoder()
		self.block_noise = self.add_noise()
		self.target = self.build_decoder()
		pass

	def build_encoder(self):
		with tf.variable_scope(name='encoder', reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.input_size, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			output = tf.layers.dense(h1, self.output_size,
			                         kernel_initializer=tf.initializers.truncated_normal)
			return output
			pass

	def add_noise(self):
		with tf.variable_scope(name='noise', reuse=tf.AUTO_REUSE):
			noise = tf.random_normal(shape=tf.shape(self.block), mean=0.0, stddev=1, dtype=tf.float32) / self.SNR
			block_noise = self.block + noise

			pass

	def build_decoder(self):
		with tf.variable_scope(name='decoder', reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.output_size, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			output = tf.layers.dense(h1, self.input_size,
			                         kernel_initializer=tf.initializers.truncated_normal)
			return output
			pass

	def build_optimizer(self):

		loss = tf.reduce_mean(tf.square(self.target - q_value))
		train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
		return loss, train_op
		pass

class Decoder(object):
	def __init__(self, env, state_size, action_size):
		self.env = env
		self.state_size = state_size
		self.action_size = action_size
		pass

	def random_action(self):
		return random.randrange(self.action_size)
		pass


if __name__ == "__main__":

	config = tf.ConfigProto()
	#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	#config.log_device_placement = False
	#config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		agent = Agent(args, sess)
		sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨
		agent.train()
		agent.save()
		agent.load()
		rewards = []
		for i in range(20):
			r = agent.play()
			rewards.append(r)
		mean = np.mean(rewards)
		print(rewards)
		print(mean)


