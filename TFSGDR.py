import tensorflow as tf
import numpy as numpy
import CartPole_QLearning_RBFNN as ql


class SGDRegressor:
	def __init__(self, dimensions, lr=1e-3):
		self.params = tf.Variable(tf.random_normal(shape=[D,1], mean=0.5, stddev=0.5), name="params")
		self.x = tf.Placeholder("float32", shape=(None, D), name="x")
		self.y = tf.Placeholder("float32",shape=(None, 1), name="x")
		linearComb = tf.matmul(x,params)
		loss = y-linearComb
		loss = tf.reduce_sum(loss * loss)

		# gradient = tf.gradients(loss, self.params)
		self.trainOp = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		self.predictOp = linearComb

		i = tf.global_variable_initializer()
		self.session = tf.InteractiveSession()
		self.session.run(init)

	def partial_fit(self, x, y):
		self.session.run(self.trainOp, feed_dict={self.x: x, self.y:y})
	def update(self, x):
		return self.session.run(self.predictOp, feed_dict={self.x=x})

if __name__ == '__main__':
	ql.SGDRegressor = SGDRegressor
	ql.main()