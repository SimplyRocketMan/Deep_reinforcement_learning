import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# the loss function we are going to use is the cross-entropy
# the function (optimzer) we are going to use to minimize cost, 
# is Adam.
# One epoch is equal to one feedforward cycle plus one backprop cycle.
# a batch 

mnist_data = input_data.read_data_sets(
	"C:/Users/ANTARTIDA/Desktop/Deep_reinforcement_learning/Basics/data",
	 one_hot=True)
# this has 10 classes, being represented as one hot 
# (one pixel is on, the rest is off [1,0,0,0] for 0 out of 3.)
# mnist dataset are images of 28x28 

n_classes = 10
batch_size = 100 
# batch_size : it makes 100 feedforwards and then backprop them.

# the xLabel will be a matrix (i.e. it has dimensions: 
# height and width.)
# the second arg in xLabel placeholder is the dimensions of 
# the matrix we are going to use. 
# [None, 784] shape tells us that the input is going to be a vector
# containing the whole image ( 28x28 = 784 ).
shape = [None, 784]
xLabel = tf.placeholder(tf.float32, [None, 784])
yLabel = tf.placeholder(tf.float32)
def conv2D(data, weights):
	return tf.nn.conv2d(data,weights,strides=[1,1,1,1], padding="SAME")

def maxPool2D(data):
	return tf.nn.max_pool(data, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def CNN_model(data): 
	weights = {"W_conv1": tf.Variable(tf.random_normal([5,5,1,32])),
				"W_conv2": tf.Variable(tf.random_normal([5,5,32,64])),
				"W_fully_connected": tf.Variable(tf.random_normal([7*7*64, 1024])),
				"out": tf.Variable(tf.random_normal([1024, n_classes]))
				} # 5x5 convolution, 1 input, 32 outputs 
	biases = {"b_conv1": tf.Variable(tf.random_normal([32])),
				"b_conv2": tf.Variable(tf.random_normal([64])),
				"b_fully_connected": tf.Variable(tf.random_normal([1024])),
				"out": tf.Variable(tf.random_normal([n_classes]))
				}

	data = tf.reshape(data, shape=[-1,28,28,1])
	conv1 = conv2D(data, weights=weights["W_conv1"])
	conv1 = maxPool2D(conv1)

	conv2 = conv2D(conv1, weights=weights["W_conv2"])
	conv2 = maxPool2D(conv2)

	fully_connected = tf.reshape(conv2, [-1,7*7*64])
	fully_connected = tf.nn.relu(tf.matmul(fully_connected, weights["W_fully_connected"]) + biases["b_fully_connected"])

	output = tf.matmul(fully_connected, weights["out"]) + biases["out"]
	return output


def backprop(xLabels):
	# the hypothesis is going to be one hot
	hypothesis = CNN_model(xLabels) 
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel))
	no_reduce_mean_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel)

	optimzer = tf.train.AdamOptimizer(0.01).minimize(cost)

	n_epochs = 50

	with tf.Session() as s:
		s.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for episode in range(int(mnist_data.train.num_examples/batch_size)):
				xLab, yLab = mnist_data.train.next_batch(batch_size) # chunks data for me 
				print("episode ", episode , " out of ", int(mnist_data.train.num_examples/batch_size))
				episode, cost_in_episode = s.run([optimzer, cost], feed_dict={xLabel: xLab, yLabel: yLab})
				epoch_loss += cost_in_episode
			print("Epoch ", epoch, "/", n_epochs, ". Loss: ", epoch_loss)

		prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(yLabel,1)) 
		# remember those are onehot

		accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
		print("Test accuracy: ", accuracy.eval({xLabel:mnist_data.test.images, 
			yLabel:mnist_data.test.labels}))

if __name__ == '__main__':
	backprop(xLabel)