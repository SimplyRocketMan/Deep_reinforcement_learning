import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym

# key:
# use gaussian distribution, and parametize the mean and variance.
# The variance can be softmax or exponential
# In this version we won't use GD, but instead we'll use hill climbing
# https://en.wikipedia.org/wiki/Hill_climbing

def make_net(Din, Dout, nnsize, output_activ_f, output_bias):
	network = []
	M1 = Din
	for n in nnsize:
		network.append(HiddenLayer(M1, n))
		M1 = n
	network.append(HiddenLayer(M1, Dout, output_activ_f, output_bias))
	return network

class HiddenLayer:
	def __init__(self, M1, M2, f=tf.nn.softmax, use_bias=True):
		self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
		self.ub = use_bias
		self.params =[self.W]
		if(use_bias):
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
			self.params.append(self.b)
		self.activ_f = f

	def forward(self, X):
		if(self.ub):
			return self.activ_f(tf.matmul(X,self.W) + self.b)
		else:
			return self.activ_f(tf.matmul(X,self.W))
"""class ValueModel:
	def __init__(self, Din, Dout, nnsize):
		self.Din = Din
		self.Dout = Dout
		self.network = make_net(Din,Dout,nnsize,lambda x:x, False)

		# TF buidlup
		self.X = tf.placeholder(tf.float32, shape=(None, Din), name='X')
		self.Y = tf.placeholder(tf.float32, shape=(None, Din), name='Y')

		# Predict op
		f = self.X
		for layer in self.network:
			f = layer.forward(f)
		self.predict_op = f

		# Train op
		loss = tf.reduce_sum(self.Y - self.predict_op)**2
		self.train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

	def set_session(self, sess):
		self.session = sess

	def predict(self, X):
		return self.session.run(self.predict_op, feed_dict={X:np.atleast_2d(X)})

	def train(self, X, Y):
		self.session.run(self.train_op, feed_dict={X:np.atleast_2d(X), Y:np.atleast_1d(Y)})"""

# Gaussian distribution, parametizing mean and variance
class PolicyModel:
	def __init__(self, Din, m_nnsize, v_nnsize):
		self.Din=Din
		self.m_nnsize = m_nnsize
		self.v_nnsize = v_nnsize

		self.network_m = make_net(Din, 1, m_nnsize, lambda x:x, False)
		self.network_v = make_net(Din, 1, v_nnsize, tf.nn.softplus, False)

		# gather params, to do hill climbing
		self.params_v_m = []
		for layer in (self.network_m + self.network_v):
			self.params_v_m += layer.params

		# tf buildups
		self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
		self.istates = tf.placeholder(tf.float32, shape=(None,Din), name='istates')
		self.advantages= tf.placeholder(tf.float32, shape=(None,), name='advantages')

		# Predict op
		f = self.istates
		for layer in self.network_m:
			f = layer.forward(f)
		self.m_predict_op = tf.reshape(f,[-1]) # mean model

		f = self.istates
		for layer in self.network_v:
			f = layer.forward(f)
		self.v_predict_op = tf.reshape(f, [-1]) # variance model

		#this was before I knew tf.contrib.distributions
		#self.predict_op = ((2*np.pi*self.v_predict_op)**-1)*
		#					tf.exp((-1/(2*self.v_predict_op))*
		#						(self.actions-self.m_predict_op)**2)		
		distribution = tf.contrib.distributions.Normal(self.m_predict_op, self.v_predict_op)
		self.predict_op = tf.clip_by_value(distribution.sample(),-1,1) # get action space for MC

		# train op
		# since we won't use minimzer, we won't use this region
	def set_session(self, sess):
		self.session = sess

	def vars_init(self):
		# init = tf.global_variables_initializer() # not useful since we won't have defined every var
		init = tf.variables_initializer(self.params_v_m)
		self.session.run(init)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={self.istates:X})

	def choose_action(self, state):
		return self.predict(state)[0]

	# hill climbing stuff
	def copy_model(self):
		ctrl_c = PolicyModel(self.Din, self.m_nnsize, self.v_nnsize)
		ctrl_c.set_session(self.session)
		ctrl_c.vars_init()
		ctrl_c.copy_model_from(self)
		return ctrl_c

	def copy_model_from(self, aself):
		ops = []
		selfparams = self.params_v_m
		otherparams= aself.params_v_m
		for q,p in zip(selfparams, otherparams):
			ret = self.session.run(p)
			op = q.assign(ret)
			ops.append(op)

		self.session.run(ops)

	def shuffling(self):
		ops = []
		for p in self.params_v_m:
			runned = self.session.run(p)
			shuffle = np.random.rand(*runned.shape) / np.sqrt(runned.shape[0])
			if np.random.rand()<0.1:
				op = p.assign(shuffle) # from scratch
			else:
				op = p.assign(runned + shuffle)
			ops.append(op)
		self.session.run(ops)

def play_one(env, policy_model, gamma):
	state = env.reset()
	totalReward = 0
	counter = 0
	done = False

	while not done and counter < 1000:
		action = policy_model.choose_action(state)
		state, reward, done, _ = env.step([action])

		totalReward += reward
		counter+=1
	return totalReward

def play_multiple_episodes(env, n_episodes, gamma, policy_model):
	rewards = np.zeros(n_episodes)

	for t in range(n_episodes):
		rewards[t] = play_one(env,policy_model,gamma)
		print("Avg until now ", rewards[:t+1].mean())
	print("Total avg ", rewards.mean())
	return rewards.mean()
def random_search(env, policy_model, gamma):
	rewards=[]
	best_r = -100000000000000;
	best_model = policy_model
	episodes_per_test = 2
	for i in range(10):
		cp_model = best_model.copy_model()
		cp_model.shuffling()
		r = play_multiple_episodes(env, episodes_per_test, gamma, cp_model)

		rewards.append(r)
		if r > best_r:
			best_model = cp_model
			print("step up in the hill ", i,"/100")
	return rewards, best_model

def main():
	env = gym.make("MountainCarContinuous-v0")
	Din = env.observation_space.shape[0]
	gamma = 0.99
	m_nnsize = [32,64,128]
	v_nnsize = [128,64,32]
	sess = tf.InteractiveSession()

	model = PolicyModel(Din,m_nnsize,v_nnsize)
	model.set_session(sess)
	model.vars_init()

	tr, model = random_search(env,model,gamma)
	print("Max reward ", np.max(tr))
	print("Min reward ", np.min(tr))

	r = play_multiple_episodes(env,75,model,gamma) # test
	print("Average test reward ",r)

	plt.plot(tr)
	plt.title("total rewards learning")
	plt.savefig("average_random_search_mcc.png")

if __name__=="__main__":
	main()
