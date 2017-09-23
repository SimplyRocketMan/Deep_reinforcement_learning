"""
22 september 2017
@author: Allan Steven Perez
@email: steven.p-m@hotmail.com
"""
import numpy as np

class FeedForwardNetwork(object):
	def __init__(self, nInputs):
		self.nInputs = nInputs
		self.vectTrain = []

	def AddLayer(self, nNeurons, activation):
		self.newLayer = HiddenLayer(nNeurons = nNeurons, activationFuncName=activation)

	def train(self, xLabels, ylabels):
		# feed_dict[]
		self.trainer = Train(xLabels, yLabels)
		# self.trainer.trainNet()
		print(xLabels)


class HiddenLayer(FeedForwardNetwork):
	def __init__(self, nNeurons ,activationFuncName = 'sigmoid'):
		super().__init__(self)
		self.nNeurons = nNeurons
		self.activationFuncName = activationFuncName
		self.activation = InitFunctions()
	def activFunc(self, x):
		self.activation.x = x
		if self.activationFuncName == 'sigmoid':
			self.activation = self.activation._i_sigmoid() 
			
		
class InitFunctions(HiddenLayer):
	def __init__(self):
		super().__init__(self)
		self.x = 0

	def _i_sigmoid(self):
		return (1+np.exp(-self.x))

	def _i_relu(self):
		if self.x > 0:
			return self.x
		else:
			return 0 

	def _i_softplus(self):
		return np.log(1+np.exp(self.x))

class Train(FeedForwardNetwork):
	def __init__(self, xLabels, yLabels):
		super().__init__(self)
		self.xLabels = xLabels
		self.yLabels = yLabels
		
	def trainNet(self):
		print('The net is training.')

	
if __name__ == '__main__':
	net = FeedForwardNetwork(2)
	net.AddLayer(32, 'sigmoid')
	net.AddLayer(128,'sigmoid')
	print(net)