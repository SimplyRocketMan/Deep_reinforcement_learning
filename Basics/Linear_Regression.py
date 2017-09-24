"""
24 September 2017
@author: Allan Perez
"""
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
	def __init__(self, trainingRate, weights, xLabels, yLabels, hypothesis):
		self.trainingRate = trainingRate
		self.weights = weights
		self.xLabels = xLabels
		self.yLabels = yLabels
		self.m = len(xLabels)
		self.hypothesis = hypothesis

	def traning(self):
		pass

	def _gradient_descent(self):
		pass

	def _cost_function(self):
		# The cost is made for each hypothesis.
		factor = 1/(2*self.m)
		diff = np.subtract(self.hypothesis, self.yLabels.T)
		diff = np.sum(diff, axis=1)
		cost = factor*diff
		return cost
	
	def _der_cost(self):
		pass

class LinearRegress:
	def _init_weights(self):
		a = self.xLabels.shape
		self.weights = np.random.rand(a[0],a[1]+1)
		self.xLabels = self.resahpe_data(self.xLabels) 


	def train(self, xLabels, yLabels,learningRate=0.001):
		self.xLabels = xLabels
		self._init_weights()
		print(xLabels,'\n\n',self.weights)
		self.predict(xLabels)
		self.trainer = Trainer(learningRate, self.weights, xLabels, yLabels, self.hypothesis)
		self.cost = self.trainer._cost_function()
		print('The cost of each hypothesis is ', self.cost)
	def resahpe_data(self, data):
		ones = np.ones((data.shape[0], data.shape[1]))
		data = np.concatenate((ones,data), axis=1)
		return data 
	def get_weights(self, weights):
		# here is suppoused to get a premade model, as a matrix, giving
		# the weights matrix a determined value
		self.weights = np.array(weights)

	def predict(self, data):
		data = self.resahpe_data(data)
		self.hypothesis = np.dot(self.weights, data.T)

if __name__ == '__main__':

	points =np.array([[1, 1],
					[2, 3],
					[4, 3],
					[3, 2],
					[5, 5]])
	
	# weights = np.array([ 0 , 1.45]).reshape(2,1)

	mine = LinearRegress()
	mine.train(points[:,0].reshape(len(points), 1), points[:,1].reshape(len(points), 1))
	prediction = mine.hypothesis
	cost = mine.cost
	
	plt.plot(points[:,0], points[:,1], 'bo')
	plt.plot(points[:,0], prediction[0], label='Prediction 1')
	plt.plot(points[:,0], prediction[1], label='Prediction 2')
	plt.plot(points[:,0], prediction[2], label='Prediction 3')
	plt.plot(points[:,0], prediction[3], label='Prediction 4')
	plt.plot(points[:,0], prediction[4], label='Prediction 5')
	plt.legend()
	plt.grid()
	plt.show()