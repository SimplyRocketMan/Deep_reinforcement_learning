import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

def StateBuild(nums): #[2,3,4,5] -> 2345
	return int("".join(map(lambda x: str(int(x)), nums)))

def Bins(x, bins): # returns the indexes of x where x is in intervals denoted by bins 
	return np.digitize(x=[x], bins=bins)[0]

class FeatureTransform:
	def __init__(self):
		self.cart_position_bins = np.linspace(-4.8,4.8,9)
		self.cart_velocity_bins = np.linspace(-3.0,3.0,9)
		self.pole_angle_bins 	= np.linspace(-0.418,0.418,9)
		self.pole_velocity_bins = np.linspace(-4.5,4.5,9)

	def Transform(self, state): 
		# Bins(cart_pos, self.cart_position_bins): 
		#	returns the index of which cart_pos belogns in np.linspace of cart_positions_bins
		# and do the same for cart_vel, pole_angle and pole_vel, and StateBuild turns the
		# [0,0,2,4] into 0024, discretizing the space of states.
		cart_pos,cart_vel,pole_angle,pole_vel = state
		return StateBuild([
			Bins(cart_pos, self.cart_position_bins), 
			Bins(cart_vel, self.cart_velocity_bins),
			Bins(pole_angle, self.pole_angle_bins),
			Bins(pole_vel, self.pole_velocity_bins),
	    ])

class Model:
	def __init__(self, env, feature_transformer):
		self.env = env
		self.feature_transformer = feature_transformer

		# 10 000 different states
		# because there are 10 spaces for each state
		num_states = 10**env.observation_space.shape[0] 
		num_actions = env.action_space.n
		self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
		# Q is a table of different possible actions in possible states.
	def Predict(self, s):
		# returns the possible actions that are available at state x
		x = self.feature_transformer.Transform(s)
		return self.Q[x]

	def Update(self, s, a, G):
		# updates the Q value at state x.
		# Gradient Descent
		x = self.feature_transformer.Transform(s)
		self.Q[x,a] += 5e-3*(G - self.Q[x,a])

	def SampleAction(self, s, epsilon):
		# returns an action, which can be random with 
		# porbability "epsilon", and the rest of the time 
		# makes the action with maximum possible return ( 0 or 1 )
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			p = self.Predict(s)
			return np.argmax(p)


def play_one(model, epsilon, gamma):
	state = env.reset()
	done = False
	totalreward = 0
	counter = 0
	while not done and counter < 10000:
		action = model.SampleAction(state, epsilon)
		prev_state = state
		state, reward, done, info = env.step(action)

		if done and counter < 199:
			reward = -450
		totalreward += reward

	    # update the model
		G = reward + gamma*np.max(model.Predict(state))
		model.Update(prev_state, action, G)

		counter += 1

	return totalreward


def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()


if __name__ == '__main__':
	ft = FeatureTransform()
	env = gym.make("CartPole-v0")
	model = Model(env, ft)
	gamma = 0.89

	N = 100000
	totalrewards = np.empty(N)
	for n in range(N):
		epsilon = 1.0/np.sqrt(n+1)
		totalreward = play_one(model, epsilon, gamma)
		totalrewards[n] = totalreward
		if n % 100 == 0:
			print("episode:", n, "total reward:", totalreward, "epsilon:", epsilon)
	print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
	print("total steps:", totalrewards.sum())

	if 'monitor' in sys.argv:
		# filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' +"GAAAMeeE"
		env = wrappers.Monitor(env, monitor_dir)

	lastReward = play_one(model, epsilon, gamma)
	
	plt.plot(totalrewards)
	plt.title("Rewards")
	plt.show()

	plot_running_avg(totalrewards)