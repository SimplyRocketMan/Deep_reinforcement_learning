"""
30 september 2017
@author: Allan Steven Perez
"""
import numpy as np

class FeedForwardNetwork():
	def __init__(self, nInputs):
		self.nInputs = nInputs
		self.vectTrain = np.zeros((nInputs,1))
		self.hiddenLayers = []
		self.weights = []

	def AddLayer(self, nNeurons, activation):
		self.hiddenLayers.append(HiddenLayer(nNeurons = nNeurons, activationFuncName=activation))

	def Train(self, xLabels, yLabels, lr=0.001):
		feed_dict = []

		self.RandomInitializenWeights()
		self.xLabels = np.array(xLabels)
		self.yLabels = np.array(yLabels)

		self.trainer = Train(xLabels, yLabels, lr)
		self.trainer.TrainNet(self)

	def RandomInitializenWeights(self):
		self.weights.append(np.random.rand(self.nInputs, self.hiddenLayers[0].nNeurons))
		for i in range(len(self.hiddenLayers)-1):
			self.weights.append(np.random.rand(self.hiddenLayers[i].nNeurons,self.hiddenLayers[i+1].nNeurons))

	def FeedForward(self):
		self.hiddenLayers[0].neurons = self.hiddenLayers[0].ActivFunc(x=np.dot(self.weights[0].T,self.xLabels ))

		for i in range(0,len(self.hiddenLayers)-1):
			try:
				self.hiddenLayers[i+1].neurons = self.hiddenLayers[i+1].ActivFunc(x
					=np.dot(self.weights[i+1].T,self.hiddenLayers[i].neurons))


			except Exception as e:	
				print(e)
				return
		return self.hiddenLayers[len(self.hiddenLayers)-1].neurons
class HiddenLayer(FeedForwardNetwork):
	def __init__(self, nNeurons ,activationFuncName = 'sigmoid'):
		self.nNeurons = nNeurons
		self.activationFuncName = activationFuncName
		self.neurons = np.ones((nNeurons, 1))

	def ActivFunc(self, x):
		if self.activationFuncName.lower() == 'sigmoid':
			return (1+np.exp(-x))**-1
		elif self.activationFuncName.lower() == 'relu':
			toReturn = []
			for i in range(len(x)):
				toReturn.append(np.max([0,x[i]]))  

			return np.array(toReturn)
		elif self.activationFuncName.lower() == 'softplus':
			return np.log(1+np.exp(x))

	def DerivativeActivFunc(self, x):
		if self.activationFuncName.lower() == 'sigmoid':
			return (np.exp(-x))/(1+np.exp(-x))**2
		elif self.activationFuncName.lower() == 'relu':
			toReturn = []
			for i in range(len(x)):
				if x[i] >=0:
					toReturn.append(1)
				else:
					toReturn.append(0)

			return np.array(toReturn)
		elif self.activationFuncName.lower() == 'softplus':
			return np.exp(x) / (1+np.exp(x))



class Train(FeedForwardNetwork):
	# The training is available only for relu activation function
	# since its derviative is the easiest computable 
	def __init__(self, xLabels, yLabels, errorFunction='squaresDifference', learningRate = 0.001):
		self.xLabels = xLabels
		self.yLabels = yLabels
		self.lr = learningRate
		
	def TrainNet(self,s):
		print('The net is training.')
		hypothesis = s.FeedForward()
		cost = self.CostFunction(hypothesis, self.yLabels)
		print("Hypothesis form Train: ", hypothesis,
			"\nLoss: ", cost)
	def derivativeWeight(self, weight):
		# only available for relu (by now)
		if weight>0:
			return 1
		else:
			return 0

	def CostFunction(self,h,y):
		# will deppend on the derivative of the 
		# activation functions of each of the 
		# layer of the NN, which are given by
		# the same class. 
		return (1/2)*np.sum((h-y)**2)

	
if __name__ == '__main__':
	xLabels = np.random.rand(2,1)
	yLabels = np.random.rand(2,1)

	net = FeedForwardNetwork(nInputs=2)
	net.AddLayer(4, 'relu')
	net.AddLayer(4,'relu')
	net.AddLayer(2,'relu')
	print("xLabels:", xLabels )
	print("yLabels:", yLabels )
	net.Train(xLabels, yLabels)