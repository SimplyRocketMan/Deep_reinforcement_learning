import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import numpy as np

mnist_data = input_data.read_data_sets(
	"C:/Users/ANTARTIDA/Desktop/Deep_reinforcement_learning/Basics/data",
	 one_hot=True)


# n_nodes = [512, 512, 256]
n_epochs = 5
n_classes = 10
batch_size = 128 

chunk_size = 28 # MNIST data are images of 28x28 dimensions.
n_chunks = 28
rnn_size = 128

xLabel = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
yLabel = tf.placeholder(tf.float32)

def RNN_model(data): 
	# layers = []

	layer = {"weights": tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 "bias": tf.Variable(tf.random_normal([n_classes]))}
	
	data = tf.transpose(data,[1,0,2])
	data = tf.reshape(data,[-1,chunk_size])
	data = tf.split(data, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size) 
	outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

	return tf.matmul(outputs[-1],layer["weights"]) + layer["bias"]

def backprop(xLabels):
	hypothesis = RNN_model(xLabels) 
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel))
	no_reduce_mean_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel)

	optimzer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as s:
		s.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for episode in range(int(mnist_data.train.num_examples/batch_size)):
				xLab, yLab = mnist_data.train.next_batch(batch_size) # chunks data for me 
				xLab = xLab.reshape((batch_size, n_chunks, chunk_size))

				episode, cost_in_episode = s.run([optimzer, cost], feed_dict={xLabel: xLab, yLabel: yLab})
				epoch_loss += cost_in_episode
			print("Epoch ", epoch, "/", n_epochs, ". Loss: ", epoch_loss)

		prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(yLabel,1)) 

		accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
		print("Test accuracy: ", accuracy.eval({xLabel:mnist_data.test.images.reshape((-1,n_chunks, chunk_size)), 
			yLabel:mnist_data.test.labels}))

if __name__ == '__main__':
	backprop(xLabel)