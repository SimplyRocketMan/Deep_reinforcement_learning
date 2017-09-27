"""
27 September 2017
@author: Allan Perez
"""
import numpy as np
# import math_functions as mf
def linearRegress(x,y):
    return np.dot(x,y)

class LinRegress:
	def __init__(self):
		pass


	def predict(self, w, x):
		return linearRegress(w,x) 

	def do(self, xLabels, yLabels, lr=0.0001):
		shape = xLabels.shape # mx1
		frame = np.zeros((shape[0],shape[1]+1))
		frame[:,1:] = xLabels
		frame[:,0] = 1
		self.frameShape = frame.shape # mxn | n=2
		weights = self._init_weights() # nx1 because I want only one hypothesis 
		trainer = Train(frame, yLabels, weights, lr)
		trainer.training()
		self.newWeights = trainer.weights # still nx1
		self.lossHist = np.array(trainer.lossPlt)
		self.weightsLst = np.array(trainer.weightsLst)
		self.weightsHistBst = self.weightsLst[np.nanargmin(self.lossHist)]
		lossBsthst = np.nanmin(self.lossHist)
		pred = self.predict(frame, self.weightsHistBst)

		return self.predict(frame, self.newWeights), pred, lossBsthst

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
		self.lossPlt = []
		self.weightsLst = []
		while np.abs(loss) > 0.01:
			# if(count == 1000000):
			# 	print("Exceeded")
			# 	return
			self.hypo = self.predict(self.xLabels, self.weights)

			alpha = self._loss()
			self.weights = self.weights - self.lr*alpha

			loss = self._loss()
<<<<<<< HEAD
			self.lossPlt.append(loss)
			self.weightsLst.append(self.weights)
			print('loss ', loss)
=======
>>>>>>> ee530614741caee32a662bf5d3d8f4a43e88b87f
			count+=1

	def _loss(self):
		m = len(self.xLabels) # number of points in the set
<<<<<<< HEAD
		toSumm = (self.hypo - self.yLabels)**2# jxn - mxn  -> jxm
=======
		toSumm = (self.hypo - self.yLabels)**2 # jxn - mxn  -> jxm
>>>>>>> ee530614741caee32a662bf5d3d8f4a43e88b87f
		loss = (1/(2*m)) * np.sum(toSumm) # jxm -> jx1 (1x1 if there's 1 hypothesis)
		return loss

	def _loss_der(self):
		m = len(self.xLabels) # number of points in the set
		toSumm = (self.hypo - self.yLabels) * self.xLabels[:,1:]# jxn - mxn  -> mxn * mxn (elementwise) -> mxn
		loss = (1/(m)) * np.sum(toSumm) # mxn -> mx1 (1x1 if there's 1 hypothesis)
		print('Sum: ',toSumm.shape) 
		return loss
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	A = np.random.rand(10, 1)
	B = np.random.rand(10, 1)

<<<<<<< HEAD
if __name__ == '__main__':
	import matplotlib.pyplot as plt

	a = np.random.rand(10,1)
	b = np.arange(0,10).reshape(10,1)

	lrg = LinRegress()
	hypo, hypo2, loss = lrg.do(b,a)

	print(loss)

	# plt.figure()
	# plt.plot(loss)


	plt.figure()
	plt.plot(b, a, 'bo-')
	plt.plot(hypo, 'r')
	plt.plot(hypo2, 'g')
	plt.show()
=======
	plt.plot(A,B)
	plt.show()
>>>>>>> ee530614741caee32a662bf5d3d8f4a43e88b87f
