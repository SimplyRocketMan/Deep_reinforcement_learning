import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

n_nodes = [256, 256, 128]
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
xLabel = tf.placeholder("float", [None, 784])
yLabel = tf.placeholder("float")

def FFNN_model(data): # CHECK THIS FUNCTIONS, IT GIVES AN ERROR INVOLVING THE LAST LAYER
	hidden_lays = []
	layers = []
	for i in range(0,len(n_nodes)):
		if i == 0: 
			hidden_lays.append(
				{"weights": tf.Variable(tf.random_normal([shape[1],
					n_nodes[i]])),
				 "bias":tf.Variable(tf.ones([n_nodes[i]])), 
				}
			)
		elif i == len(n_nodes)-1:
			hidden_lays.append(
				{"weights": tf.Variable(tf.random_normal([n_nodes[i],
					n_classes])),
				 "bias":tf.Variable(tf.ones([n_classes])), 
				}
			)
		else:
			hidden_lays.append(
				{"weights": tf.Variable(tf.random_normal([n_nodes[i-1],
					n_nodes[i]])),
				 "bias":tf.Variable(tf.ones([n_nodes[i]])), 
				}
			)
	for i in range(len(hidden_lays)):
		if i == 0: 
			layers.append(tf.nn.relu(tf.add(tf.matmul(
				data, hidden_lays[i]["weights"]), 
				hidden_lays[i]["bias"])))
		else:
			layers.append(tf.nn.relu(tf.add(tf.matmul(
				layers[i-1], hidden_lays[i]["weights"]), 
				hidden_lays[i]["bias"])))

	return layers[len(layers)-1]


def backprop(xLabels):
	# the hypothesis is going to be one hot
	hypothesis = FFNN_model(xLabels) 
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel))
	no_reduce_mean_cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=yLabel)

	optimzer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 20

	with tf.Session() as s:
		s.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for episode in range(int(mnist_data.train.num_examples/batch_size)):
				xLab, yLab = mnist_data.train.next_batch(batch_size) # chunks data for me 
				episode, cost_in_episode = s.run([optimzer, cost], feed_dict={xLabel: xLab, yLabel: yLab})
				epoch_loss += cost_in_episode
			print("Epoch ", epoch, "/", n_epochs, ". Loss: ", epoch_loss)

		prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(yLabel,1)) 
		# remember those are onehot

		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		print("Test accuracy: ", accuracy.eval({xLabel:mnist_data.test.images, 
			yLabel:mnist_data.test.labels}))

if __name__ == '__main__':
	backprop(xLabel)