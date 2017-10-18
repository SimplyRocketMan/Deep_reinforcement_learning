import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

n_epochs = 100
n_episodes = 100


	# state = env.reset() # starting state -> array

	# box = env.observation_space

	# ActionSpace = env.action_space

	# env.step() returns:
	# "info" is for debugging	
	# obs -> observation (I guess it is the state)
# done -> flag
def TakeAction(s,w):
	return 1 if s.dot(w) > 0.01 else 0
def Play(env, params):
	counter = 0
	state = env.reset()
	done = False
	while not done and counter <10000:
		# env.render() # to see the play
		action = TakeAction(state, params)
		counter+=1
		state, reward, done, info = env.step(action) # params can be only 0 or 1
	return counter

def MultiplePlays(env, weights, returnList=False):
	episodesPlayed = []
	for i in range(n_episodes):
		episodesPlayed.append(Play(env, weights))
	if returnList == True:
		return episodesPlayed
	else:
		return np.mean(episodesPlayed)
def AdjustWeights(env):
	weights = []
	best_so_far = 0
	eLHist = []
	for i in range(n_epochs):
		
		new_weights = np.random.random(4,)*2 - 1
		eL = MultiplePlays(env, new_weights)
		eLHist.append(eL)
		print("Epoch #",i," out of ",n_epochs, " Avg len of game: ",eL)
		if(eL > best_so_far):
			weights = new_weights
			best_so_far = eL#np.max(eLHist)
	return weights, eLHist
	

if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	env.reset()
	weights,eLHist = AdjustWeights(env)
	# play last time
	print(weights)
	env = wrappers.Monitor(env, "c:/Users/ANTARTIDA/Desktop/Deep_reinforcement_learning/video/game")
	l = MultiplePlays(env, weights,True)
	plt.figure(str(np.mean(l)))
	plt.plot(l)
	plt.figure()
	plt.plot(eLHist)
	plt.show()