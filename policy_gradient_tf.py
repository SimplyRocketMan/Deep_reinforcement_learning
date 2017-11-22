import tensorflow as tf
import numpy as np
import gym 
import os 

from gym import wrappers
from datetime import datetime


class HiddenLayer:
	def __init__(self, M1, M2, activ_f=tf.nn.tanh, use_bias=True):
		self.W = tf.Variable(tf.random_normal(shape=(M1,M2)))
		#self.params = [self.W]
		self.use_bias = use_bias
		if(use_bias):
			self.b = tf.Variable(np.zero(M2).astype(np.float32))
			#params.append(self.b)
		self.activ_f = activ_f
	def feed_forward(self, X):
		X = np.array(X)
		if(self.use_bias):
			return activ_f(tf.matmul(self.W, X) + self.b)
		else:
			return activ_f(tf.matmul(self.W, X))

# V(s) approximator
class ValueModel:
	def __init__(self, D_in, D_out, NN_size):
		self.Din = D_in
		self.Dout = D_out # n actions
		self.layers = []
		# FFNN 
		M1 = D_in
		for M2 in NN_size:
			Layer = HiddenLayer(M1, M2)
			M1 = M2
			self.layers.append(Layer)
		self.layers.append(HiddenLayer(M1, D_out, lambda x:x, False))

		self.X = tf.placeholder(tf.float32, shape=(None,D_in), name='X')
		self.Y = tf.placeholder(tf.int32, shape=(None, ), name='Y')

		forward = self.X
		for i in self.layers:
			forward = i.feed_forward(forward)
		prediction = forward
		self.ff_op = prediction

		loss = tf.reduce_sum(tf.square(self.Y - self.ff_op))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

	def set_session(self, sess):
		self.session = sess

	def predict(self, X):
		return self.session.run(self.ff_op, feed_dict={self.X:X})

	def partial_fit(self, X, Y ):
		self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})

# pi(a|s) approximator
class PolicyModel:
	def __init__(self, D_in, D_out, NN_size):
		self.Din = D_in
		self.Dout = D_out
		self.layers = []
		# FFNN 
		M1 = D_in
		for M2 in NN_size:
			Layer = HiddenLayer(M1, M2)
			M1 = M2
			self.layers.append(Layer)
		self.layers.append(HiddenLayer(M1, D_out, tf.nn.softmax, False))

		self.state_input = tf.placeholder(tf.float32, shape=(None,D_in), name='state_input')
		self.action = tf.placeholder(tf.int32, shape=(None, ), name='action')
		self.advantage = tf.placeholder(tf.float32, shape=(None, ), name='advantage')

		forward = self.state_input
		for i in self.layers:
			forward = i.feed_forward(forward)
		pi_a_given_s = forward
		self.ff_op = pi_a_given_s

		probabilities = tf.log(tf.reduce_sum(pi_a_given_s*tf.one_hot(self.action, D_out), axis=[1]))

		inverse_performace = -tf.reduce_sum(self.advantage * probabilities)
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(inverse_performace) # goal: maximize performance

	def set_session(self, sess):
		self.session = sess

	def predict(self, X):
		return self.session.run(self.ff_op, feed_dict={self.state_input:X})

	def partial_fit(self, Xs, actions, advantages ):
		self.session.run(self.train_op, feed_dict={self.X:Xs, self.action:actions, self.advantage:advantages})

	def choose_action(self, X):
		probabilities = self.predict(X)
		return np.random.choice(len(probabilities), p=probabilities) 

def play_one_td(env, policy_model, value_model, gamma):
	state = env.reset()
	totalReward = 0
	counter = 0
	done = False

	while not done and counter < 1000:
		action = policy_model.choose_action(state)
		prev_state =state
		state, reward, done, _ = env.step(action)

		if(done):
			reward = -500

		V_t1 = value_model.predict(state)
		V_t  = value_model.predict(prev_state)
		G = reward +gamma*np.max(V_t1)
		advantage = G - V_t
		policy_model.partial_fit(prev_state, action, advantage)
		value_model.partial_fit(prev_state, G)

		totalReward += reward
		counter+=1
	return totalReward

def Main():
	env = gym.make("CartPole-v0")
	D_in = env.observation_space.shape[0]
	D_out = env.action_space.n
	policy_model = PolicyModel(D_in, D_out, [16])
	value_model = ValueModel(D_int,D_out,[16,16])
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())
	policy_model.set_session(session)
	value_model.set_session(session)
	gamma = 0.8

	if('monitor' in sys.argv):
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	N = 1000
	totalReward = []
	loss = []

	for i in range(N):
		totalreward = play_one_td(env, policy_model, value_model, gamma)
		totalReward.append(totalreward)
		if n % 100 == 0:
			print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", 
				np.array(totalReward)[max(0, i-100):(i+1)].mean())

	print("avg reward for last 100 episodes:", np.array(totalReward)[-100:].mean())
	print("total steps:", np.array(totalReward).sum())
