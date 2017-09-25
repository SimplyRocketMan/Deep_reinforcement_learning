"""
24 September 2017
@author: Allan Perez
"""
# problem -> divergence 
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
	def __init__(self, learningRate, weights, xLabels, yLabels, hypothesis):
		self.learningRate = learningRate
		self.weights = weights
		self.xLabels = xLabels
		self.yLabels = yLabels
		self.m = len(xLabels)
		self.hypothesis = hypothesis

	def training(self):
		self._gradient_descent()

	def _gradient_descent(self):
		self.get_best_hypothesis()
		self.cost_deriv = self._der_cost() 
		self.xLabels = self.resahpe_data(self.xLabels)
		print('hypothesis: ',self.hypothesis)
		print('weights: ', self.weights)
		print('best index: ', self.best_hypo_index)
		print('cost der: ', self.cost_deriv)
		print('cost: ', self.cost)
		print(self.xLabels.shape)
		while np.abs(self.cost) > 0.001:
			if(np.abs(self.cost) > 50):
				print('We\'ve reached a divergence')
				self.weights = self.initial_weights
				return
			self.weights = np.array(self.weights - self.learningRate*self.cost_deriv)
			self.predict()
			self._cost_function()
			self._der_cost()
			print(self.cost)
		print('The new weights are: ', self.weights)
		print('The last hypothesis is ', self.hypothesis)


	def _cost_function(self):
		# The cost is made for each hypothesis.
		factor = 1/(2*self.m)
		diff = np.subtract(self.hypothesis, self.yLabels.T) 
		diff = np.sum(diff, axis=1)
		self.cost = factor*diff
		return self.cost
	
	def _der_cost(self):
		der_factor = 1/(self.m)
		der_diff = np.multiply(np.subtract(self.hypothesis, self.yLabels.T), self.xLabels.T)
		der_diff = np.sum(der_diff, axis=1)
		self.der_cost = der_factor*der_diff
		return self.der_cost

	def get_best_hypothesis(self):
		self.best_hypo_index = (np.abs(self.cost-0)).argmin()
		self.cost = self.cost[self.best_hypo_index]
		self.hypothesis = self.hypothesis[self.best_hypo_index]
		self.initial_weights = self.weights[self.best_hypo_index]
		self.weights = self.weights[self.best_hypo_index]

	def predict(self):
		self.hypothesis = np.dot(self.weights, self.xLabels.T)
		return self.hypothesis

	def resahpe_data(self, data):
		ones = np.ones((data.shape[0], data.shape[1]))
		data = np.concatenate((ones,data), axis=1)
		return data 

class LinearRegress:
	def _init_weights(self):
		a = self.xLabels.shape
		self.weights = np.random.rand(a[0],a[1]+1)
		self.xLabels = self.resahpe_data(self.xLabels) 


	def train(self, xLabels, yLabels,learningRate=0.0001):
		print(xLabels)
		self.xLabels = xLabels
		self._init_weights()
		self.predict(xLabels)
		self.trainer = Trainer(learningRate, self.weights, xLabels, yLabels, self.hypothesis)
		self.cost = self.trainer._cost_function()
		self.der_cost = self.trainer._der_cost()
		self.trainer.training()
		self.weights = self.trainer.weights
		print('The cost of each hypothesis is ', self.cost)
		print('Derivative of the cost: ', self.der_cost)
	def resahpe_data(self, data):
		ones = np.ones((data.shape[0], data.shape[1]))
		data = np.concatenate((ones,data), axis=1)
		print(data)
		return data 
	def get_weights(self, weights):
		# here is suppoused to get a premade model, as a matrix, giving
		# the weights matrix a determined value
		self.weights = np.array(weights)

	def predict(self, data):
		data = self.resahpe_data(data)
		self.hypothesis = np.dot(self.weights, data.T)

	def _predict(self, data):
		data = self.resahpe_data(data)
		hypothesis = np.dot(self.weights.T, data.T)
		return hypothesis

if __name__ == '__main__':

	A = np.random.uniform(0,40,20).reshape(20,1)
	B = np.random.uniform(0,40,20).reshape(20,1)
	points=np.concatenate((A,B), axis=1)
	# weights = np.array([ 0 , 1.45]).reshape(2,1)

	mine = LinearRegress()
	mine.train(points[:,0].reshape(len(points), 1), points[:,1].reshape(len(points), 1))
	prediction = mine.hypothesis
	prediction_ = mine._predict(points[:,0].reshape(len(points), 1))
	print(prediction_)
	print('Last loss: ', mine.trainer.cost)

	cost = mine.cost

	plt.plot(points[:,0], points[:,1], 'bo')
	plt.plot(points[:,0], prediction_, label='Trained Prediction')
	plt.legend()
	plt.grid()
	plt.show()