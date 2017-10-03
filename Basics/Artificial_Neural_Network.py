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

	def Train(self, xLabels, ylabels):
		# feed_dict[]
		# self.trainer.trainNet()
		self.RandomInitializenWeights()
		self.xLabels = np.array(xLabels)
		print("Input layer: ", self.xLabels.shape)
		print("Weights: ", self.weights[0].shape)

		self.hiddenLayers[0] = self.hiddenLayers[0].ActivFunc(x=np.dot(self.weights[0],self.xLabels))
		print("Hidden layer ",0,self.hiddenLayers[0])
		print("Len", len(self.hiddenLayers))
		for i in range(0,len(self.hiddenLayers)-1):
			try:
				self.hiddenLayers[i+1] = self.hiddenLayers[i+1].ActivFunc(x=np.dot(self.weights[i+1].T,self.hiddenLayers[i]))
				print("Hidden layer ",i+1,self.hiddenLayers[i+1])

			except Exception as e:	
				print(e)
				return
		print(len(self.hiddenLayers))
	def RandomInitializenWeights(self):
		self.weights.append(np.random.rand(self.nInputs, self.hiddenLayers[0].nNeurons))
		for i in range(len(self.hiddenLayers)-1):
			self.weights.append(np.random.rand(self.hiddenLayers[i].nNeurons,self.hiddenLayers[i+1].nNeurons))

class HiddenLayer(FeedForwardNetwork):
	def __init__(self, nNeurons ,activationFuncName = 'sigmoid'):
		self.nNeurons = nNeurons
		self.activationFuncName = activationFuncName
		self.neurons = np.ones((nNeurons, 1))

	def ActivFunc(self, x):
		if self.activationFuncName.lower() == 'sigmoid':
			return (1+np.exp(-x))**-1
		elif self.activationFuncName.lower() == 'relu':
			if x >0:
				return x  
			else:
				return 0
		elif self.activationFuncName.lowe() == 'softplus':
			return np.log(1+np.exp(x))



class Train(FeedForwardNetwork):
	def __init__(self, xLabels, yLabels):
		super().__init__(self)
		self.xLabels = xLabels
		self.yLabels = yLabels
		
	def TrainNet(self):
		print('The net is training.')

	
if __name__ == '__main__':
	xLabels = np.random.randint(0,10,(2,1))
	yLabels = np.random.randint(0,10,(2,1))

	net = FeedForwardNetwork(2)
	net.AddLayer(2, 'sigmoid')
	net.AddLayer(4,'sigmoid')
	net.AddLayer(4,'sigmoid')


	net.Train(xLabels, yLabels)