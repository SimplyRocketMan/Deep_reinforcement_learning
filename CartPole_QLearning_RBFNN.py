import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from MountainCar import plot_running_avg

# observation_examples = np.random.rand((20000,4))*2 -2 

class FeatureTransformer():
	def __init__(self, env, n_components=2000):
		observation_examples = np.random.rand(20000,4)*2 -2 
		scaler = StandardScaler()
		scaler.fit(observation_examples)

		feature_machine = FeatureUnion([
			("RBFK1", RBFSampler(gamma=pow(2.0,3), n_components=n_components)),
			("RBFK2", RBFSampler(gamma=pow(2.0,2), n_components=n_components)),
			("RBFK3", RBFSampler(gamma=pow(2.0,1/3), n_components=n_components)),
			("RBFK4", RBFSampler(gamma=pow(2.0,1/2), n_components=n_components)),
			("RBFK5", RBFSampler(gamma=pow(2.0,1/5), n_components=n_components))
			])
		example_features = feature_machine.fit_transform(
			scaler.transform(observation_examples))

		self.dimensions = example_features.shape[1]
		self.scaler = scaler
		self.feature_machine = feature_machine

	def transform(self, state):
		scaled = self.scaler.transform(state)
		return self.feature_machine.transform(scaled)

class SGDRegressor():
	# this is a gradient descent of a linear model.
	# w := w-lr*Gradient(Loss(w))
	def __init__(self, v):
		self.w = np.random.rand(v) / v**(1/2)
		self.lr = 1e-1

	def partial_fit(self,X,Y):
		self.w+= self.lr*np.dot(Y-np.dot(X,self.w),X)

	def predict(self, X):
		return X.dot(self.w)

class Model():
	def __init__(self, env, feature_transformer):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			model = SGDRegressor(feature_transformer.dimensions)
			model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
			self.models.append(model)

	def predict(self, state):
		X = self.feature_transformer.transform(np.array([state]))
		assert(len(X.shape) == 2)
		return np.array([m.predict(X)[0] for m in self.models])

	def update(self, state, action,Gvalue):
		X = self.feature_transformer.transform(np.array([state]))
		assert(len(X.shape) == 2)
		self.models[action].partial_fit(X, [Gvalue])
	def choose_action(self, state, epsilon):
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(state))
def play_episode(model, env, epsilon, gamma):
	state = env.reset()
	flag = False
	totalReward = 0 
	counter =0
	while not flag and counter < 800:
		action = model.choose_action(state, epsilon)
		stateT = state
		state, reward, flag, info = env.step(action)

		G = reward + gamma*np.max(model.predict(state))
		model.update(stateT,action,G)

		totalReward += reward
		counter+=1
	return totalReward

if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	machine = FeatureTransformer(env)
	model = Model(env, machine)
	gamma = 0.99

	Episodes = 301
	totalReward = []
	for i in range(Episodes):
		epsilon = 1.0/np.sqrt(i+1)
		totalreward = play_episode(model, env, epsilon, gamma)
		totalReward.append(totalreward)
		if i%50 == 0:
			print("episode:", i, "total reward:", totalreward, "epsilon:", epsilon)
	print("avg reward for last 100 episodes:", np.array(totalReward)[-100:].mean())
	print("total steps:", np.array(totalReward).sum())
		
	if 'monitor' in sys.argv:
		# filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' +"CartPoleGameVideo"
		env = wrappers.Monitor(env, monitor_dir)

	lastRew = play_episode(model, env, 0, gamma)

	plt.plot(totalReward)
	plt.title("Rewards")
	plt.show()

	plot_running_avg(np.array(totalReward))