"""
27 September 2017
@author: Allan Perez
"""
import numpy as np
from math_functions import linearRegress, logisticRegress	


class Regress:
	def __init__(self):
		pass

	def do(self, xLabels, yLabels, trainingType, lr=0.001):
		shape = xLabels.shape # mx1
		self.frame = np.zeros((shape[0],shape[1]+1))
		self.frame[:,1:] = xLabels
		self.frame[:,0] = 1
		self.frameShape = self.frame.shape # mxn | n=2
		self.lr = lr
		self.yLabels = yLabels
		self.trainingType = trainingType

		if trainingType.lower()=="linear": 
			self.linearTrain()
			return linearRegress(self.newWeights.T,self.frame.T) , self.newWeights
		elif trainingType.lower()=="logistic": 
			print(self.frame)
			self.logisticTrain()
			return logisticRegress(self.newWeights.T,self.frame.T), self.newWeights

		
	def _init_weights(self):
		# return np.random.rand(self.frameShape[1], 1)
		return np.random.rand(self.frameShape[1], 1) / np.sqrt(self.frameShape[1])

	def linearTrain(self):
		weights = self._init_weights() # nx1 because I want only one hypothesis
		trainer = LinearTrain(self.frame, self.yLabels, weights, self.lr, self.trainingType)
		trainer.training()
		self.newWeights = trainer.weights # still nx1

	def logisticTrain(self):
		weights = self._init_weights() # nx1 because I want only one hypothesis
		trainer = LogisticTrain(self.frame, self.yLabels, weights, self.lr, self.trainingType)		
		trainer.training()
		print("suppoused to be trainning")
		self.newWeights = trainer.weights # still nx1

class LinearTrain():
	def __init__(self, xLabels, yLabels, weights, lr, trainingType):
		# super(Regress, self).__init__()
		self.xLabels = xLabels # mxn
		self.yLabels = yLabels # mx1
		self.weights = weights  # nx1 
		self.trainingType = trainingType
		self.lr = lr
		self.choose()


	def training(self):
		loss = self._loss()
		count = 0
		print("Weights: ", self.weights )

		while np.abs(loss) > 0.0001:
			if(count == 10000):
				print("Exceeded")
				return
			self.choose()
			# print("Hypothesis: ",self.hypo)
			# print("yLabels: ",self.yLabels)
			# print("Subtraciton: ",self.hypo -self.yLabels)
			# print("Squared: ",(self.hypo -self.yLabels)**2)
			# print("Summed: ",np.sum((self.hypo -self.yLabels)**2))
			# print("Loss: ", loss)
			# print("loss suppoused: ",1/10*np.sum((self.hypo -self.yLabels)**2))
			alpha = self._loss_der()
			
			self.weights = self.weights - self.lr*alpha
	
			loss = self._loss()
			print("LOSS: ",loss)
			count+=1

	def _loss(self):
		m = len(self.xLabels) # number of points in the set
		toSumm = (self.hypo - self.yLabels)**2 # jxn - mxn  -> jxm
		loss = (1/(m)) * np.sum(toSumm) # jxm -> jx1 (1x1 if there's 1 hypothesis)
		return loss

	def _loss_der(self):
		m = len(self.xLabels) # number of points in the set
		toSumm = (self.hypo - self.yLabels) # jxn - mxn  -> mxn * mxn (elementwise) -> mxn
		loss = (1/(m)) * np.sum(toSumm* (self.xLabels[:,1:])) # mxn -> mx1 (1x1 if there's 1 hypothesis)
		return loss

	def choose(self):
		if self.trainingType.lower() ==  "linear":
			self.hypo = linearRegress(self.xLabels, self.weights)
		elif self.trainingType.lower() ==  "logistic":
			self.hypo = logisticRegress(self.xLabels, self.weights)

class LogisticTrain(LinearTrain):
	def _loss(self):
		m = len(self.xLabels)

		firstPart  = self.yLabels * np.log(self.hypo)
		secondPart = (1-self.yLabels)*np.log(1-self.hypo)
		toSum = firstPart + secondPart
		return (-1/m )*np.sum(toSum)

	def _loss_der(self):
		toSumm = self.hypo - self.yLabels
		toSumm = toSumm * self.xLabels
		return np.sum(toSumm)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# X = np.random.uniform(low=0, high=100, size=10).reshape(10,1)
	# Y = -X + 1 + np.random.normal(scale=10, size=10).reshape(10,1)

	# K = Regress()
	# pred = K.do(X,Y, "linear")

	# plt.plot(X,Y, 'bo')
	# plt.plot(X, pred)
	# plt.show()
	# def sigmoid(x):
	# 	return (1+np.exp(-x))**-1

	N = 100
	D = 2	
	
	X = np.random.randn(N,D)

	# center the first 50 points at (-2,-2)
	X[:50,:] = X[:50,:] - 2*np.ones((50,D))

	# center the last 50 points at (2, 2)
	X[50:,:] = X[50:,:] + 2*np.ones((50,D))

	X = X.reshape(100,2,1)
	fr = []

	for i in X:
		if i[0] >=0:
			fr.append(1)
		else:
			fr.append(0)
	fr = np.array(fr).reshape(100,1)
	lgr = Regress()
	hypo, weights = lgr.do(X[:,0], fr, trainingType='logistic', lr=0.0000001)

	print(hypo, hypo.shape, weights)

	# plt.plot(X[:,0],X[:,1], 'bo')
	plt.plot(X[:,0], hypo[0], 'r')
	plt.plot(X[:,0], fr, 'bo')
	plt.show()