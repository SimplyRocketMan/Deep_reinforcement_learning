"""
27 September 2017
@author: Allan Perez
"""
import numpy as np
from math_functions import linearRegress


class LinRegress:
	def __init__(self):
		pass


	def predict(self, w, x):
		return linearRegress(w,x) 

	def do(self, xLabels, yLabels, lr=0.0005):
		shape = xLabels.shape # mx1
		frame = np.zeros((shape[0],shape[1]+1))
		frame[:,1:] = xLabels
		frame[:,0] = 1
		self.frameShape = frame.shape # mxn | n=2
		weights = self._init_weights() # nx1 because I want only one hypothesis 
		trainer = LinearTrain(frame, yLabels, weights, lr)
		trainer.training()
		self.newWeights = trainer.weights # still nx1

		return self.predict(frame, self.newWeights)

	def _init_weights(self):
		# return np.random.rand(self.frameShape[1], 1)
		return np.random.rand(self.frameShape[1], 1) / np.sqrt(self.frameShape[1])

class LinearTrain():
	def __init__(self, xLabels, yLabels, weights, lr):
		# super(LinRegress, self).__init__()
		self.xLabels = xLabels # mxn
		self.yLabels = yLabels # mx1
		self.weights = weights  # nx1 
		self.hypo = linearRegress(self.xLabels, self.weights)
		self.lr = lr

	def training(self):
		loss = self._loss()
		count = 0
		while np.abs(loss) > 0.0001:
			if(count == 10000):
				print("Exceeded")
				return
			self.hypo = linearRegress(self.xLabels, self.weights)
			# print("Hypothesis: ",self.hypo)
			# print("yLabels: ",self.yLabels)
			# print("Subtraciton: ",self.hypo -self.yLabels)
			# print("Squared: ",(self.hypo -self.yLabels)**2)
			# print("Summed: ",np.sum((self.hypo -self.yLabels)**2))
			# print("Loss: ", loss)
			# print("loss suppoused: ",1/10*np.sum((self.hypo -self.yLabels)**2))
			alpha = self._loss_der()
			
			self.weights = self.weights - self.lr*alpha
			print(self.weights,alpha) 
	
			loss = self._loss()
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
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	X = np.random.uniform(low=0, high=100, size=10).reshape(10,1)
	Y = -X + 1 + np.random.normal(scale=10, size=10).reshape(10,1)

	K = LinRegress()
	pred = K.do(X,Y)

	plt.plot(X,Y, 'bo')
	plt.plot(X, pred)
	plt.show()
