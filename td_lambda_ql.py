import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import CartPole_QLearning_RBFNN as qL
from CartPole_QLearning_RBFNN import FeatureTransformer, plot_running_avg, Model
from TFSGDR import SGDRegressor

# G(n, t) = R(t+1)+g*R(t+2)+...+g^n-1*R(t+n)+g^n*V(s(t+n))
qL.SGDRegressor = SGDRegressor

def play_one(model, epsilon, gamma, Lambda):
	state = env.reset()
	done = False
	rewards = 0
	count = 0
	while not done and count <1000:
		action = model.choose_action(state, epsilon)
		stateT = state
		state, reward, done, _=env.step(action)

		G = reward + gamma*np.max(model.predict(state)[0])
		model.update(stateT, action, G, gamma, Lambda)

		rewards+=reward
		count+=1

	return rewards

def SGDElegibility: # linear model
	def __init__(self, Din):
		self.param = np.random.rand(Din)/np.sqrt(Din/2)
	def partial_fit(self, X, Y, e, lr=1e-3):
		self.param += lr*(Y-predict(X)) * e 
	def predict(self, X):
		return np.array(X).dot(self.params)

class Model:
	def __init__(self, env, FeatureTransformer):
		self.env = env
		self.models = []
		self.FT = FeatureTransformer

		D = FeatureTransformer.dimensions
		self.e = np.zeros((env.action_space.n, D))
		for i in range(env.action_space.n):
			model = SGDElegibility(D)
			self.models.append(model)


	def predict(self, state):
		X = self.FT.transform([state])
		assert(len(X.shape)==2)
		return np.array([m.predict(x)[0] for m in self.models])

	def update(self, state, action, G, gamma, Lambda):
		X = self.FT.transform([state])
		assert(len(X.shape)==2)
		self.e = self.e*gamma*Lambda
		self.e[action] += X[0]
		self.models[action].partial_fit(X[0], G, self.e[action])

	def choose_action(self, epsilon, state):
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(state)) 



if __name__ == '__main__':
	env = gym.make("MountainCar-v0")
	ft = FeatureTransformer(env)
	model = Model(env, ft)
	gamma = 0.99
	Lambda = 0.7

	if 'monitor' in sys.argv:
		file = "./awsomeTDLearning"
		env = wrappers.Monitor(env, file)

	Episodes = 301
	totalReward = []
	for i in range(Episodes):
		epsilon = 1.0/np.sqrt(i+1)
		totalreward = play_episode(model, epsilon, gamma, env)
		totalReward.append(totalreward)
		print("episode:", i, "total reward:", totalreward, "epsilon:", epsilon)

	print("avg reward for last 100 episodes:", np.array(totalReward)[-100:].mean())
	print("total steps:", np.array(totalReward).sum())