"""
27 September 2017
@author: Allan Perez
"""
import numpy as np
import math_functions as mf

class LinRegress:
	def __init__(self):
		pass


	def predict(self, w, x):
		return mf.linearRegress(w,x) 

	def do(self, xLabels, yLabels, lr=0.001):
		shape = xLabels.shape # mx1
		frame = np.zeros((shape[0],shape[1]+1))
		frame[:,1:] = xLabels
		frame[:,0] = 1
		self.frameShape = frame.shape # mxn | n=2
		weights = self._init_weights() # nx1 because I want only one hypothesis 
		trainer = Train(frame, yLabels, weights, lr)
		trainer.training()
		self.newWeights = trainer.weights # still nx1

		return self.predict(frame, self.newWeights)

	def _init_weights(self):
		# return np.random.rand(self.frameShape[1], 1)
		return np.random.rand(self.frameShape[1], 1) / np.sqrt(self.frameShape[1])

class Train(LinRegress):
	def __init__(self, xLabels, yLabels, weights, lr):
		super(LinRegress, self).__init__()
		self.xLabels = xLabels # mxn
		self.yLabels = yLabels # mx1
		self.weights = weights  # nx1 
		self.hypo = self.predict(self.xLabels, self.weights)
		self.lr = lr

	def training(self):
		loss = self._loss()
		count = 0
		while np.abs(loss) > 0.0001:
			if(count == 1000000):
				print("Exceeded")
				return
			self.hypo = self.predict(self.xLabels, self.weights)

			alpha = self._loss()
			self.weights = self.weights - self.lr*alpha

			loss = self._loss()
			print('loss ', loss)
			count+=1

	def _loss(self):
		m = len(self.xLabels) # number of points in the set
		toSumm = self.hypo - self.yLabels # jxn - mxn  -> jxm
		loss = (1/(2*m)) * np.sum(toSumm) # jxm -> jx1 (1x1 if there's 1 hypothesis)
		return loss

	def _loss_der(self):
		m = len(self.xLabels) # number of points in the set
		toSumm = (self.hypo - self.yLabels) * self.xLabels[:,1:]# jxn - mxn  -> mxn * mxn (elementwise) -> mxn
		loss = (1/(m)) * np.sum(toSumm) # mxn -> mx1 (1x1 if there's 1 hypothesis)
		print('Sum: ',toSumm.shape)
		return loss

