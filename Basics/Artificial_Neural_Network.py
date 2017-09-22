"""
22 september 2017
@author: Allan Steven Perez
@email: steven.p-m@hotmail.com
"""
import numpy as np

class FeedForwardNetwork:
	def __init__(self, nInputs):
		self.nInputs = nInputs
		self.vectTrain = []

	def init_weights(self, l1, l2):
		print('init weights')

class HiddenLayer(FeedForwardNetwork):
	def __init__(self, nNeurons ,activationFuncName = 'sigmoid'):
		super().__init__(self)
		self.nNeurons = nNeurons
		self.activationFuncName = activationFuncName
		self.activation = InitFunctions()
	def activFunc(self):
		if self.activationFuncName == 'sigmoid':
			self.activation = self.activation._i_sigmoid(self.vectTrain) 
			
		
class InitFunctions(HiddenLayer):
	def __init__(self, x):
		super().__init__()
		self.x = x

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
	def __init__(self, inputVect):
		super().__init__(self)
		self.vectTrain = inputVect

		