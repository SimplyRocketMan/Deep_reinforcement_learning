
import theano
import theano.tensor as T
from theano import function
import CartPole_QLearning_RBFNN
import numpy

class SGDRegressor:
	def __init__(self, dimensions):
		print("Started21111")
		params = np.random.rand(dimensions)
		self.params = params
		self.lr = 1e-3

		x = T.dmatrix("x")
		y = T.dvector("y")
		linearComb = x.dot(params)
		loss = (y - linearComb ) ** 2
		grad = T.grad(loss, params)

		self.partial_fit = function([x,y], self.weights -self.lr * grad)
		self.predict = function([x], linearComb)

	def partial_fit(self,xx,yy):
		self.weights = self.partial_fit(xx,yy)

	def predict(self,xx):
		return self.predict(xx)

if __name__ == '__main__':
	print("Started")
	CartPole_QLearning_RBFNN.SGDRegressor = SGDRegressor
	CartPole_QLearning_RBFNN.main()

