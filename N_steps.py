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

def play_one(model, epsilon, gamma, n_steps):
	state = env.reset()
	done = False
	rewardSum = 0
	rewardHist = []
	states = []
	actions = []
	iterations = 0

	G_param = np.array([gamma]*n_steps)**np.arrange(n_steps)

	while not done and iterations < 1000:
		action = model.sample_action(state, epsilon)

		states.append(state)
		actions.append(action)

		stateT = state
		state, reward, done, _ = env.step(action) 

		rewardHist.append(reward)
		states.append(state)
		actions.append(action)

		rewardSum+=reward


		if(n_steps % iterations == 0): # if n steps, do the stuff
			G = np.sum(G_param*rewardHist) + gamma**n_steps*np.max(model.predict(state))
			model.update(states,actions, G)
			states = []
			actions= []
			rewardHist = []
		iterations += 1

	return rewardSumd
if __name__ == '__main__':
	print("Hey there, it started")
	env = gym.make("MountainCar-v0")
	f = FeatureTransformer(env)
	model = Model(env,f)
	gamma = 0.99
	n_steps = 5

	if 'monitor' in sys.argv:
		file = "C:/Users/ANTARTIDA/Desktop/Deep_reinforcement_learning/n_step_video/awsomevid"
		env = wrappers.Monitor(env, file)
	print("Second phase")
	steps = 300
	rewards = []
	for i in range(N):
		epsilon = 1/np.sqrt(i+1)
		r = play_one(model,epsilon,gamma,n_steps)
		rewards.append(r)
		print("episode ",i," r = ",r)
	print("Avg reward for last 200 episodes ", np.array(rewards)[-200:].mean())
	plot_running_avg(np.array(rewards))