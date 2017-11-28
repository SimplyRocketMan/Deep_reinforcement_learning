import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize

env = gym.make("Breakout-v0")
A = env.reset()
B = A[31:195] # get only the data that we need 
C = imresize(B, size=(105, 80, 3)) # fancy
C2 = imresize(B, size=(105, 80, 3), interp='nearest')# nearest neighbour
CS = imresize(B, size=(80, 80, 3), interp="nearest")# square

plt.imsave("breakout_resize", C2) # save the image

tf.contrib.layers.conv2d # convlutional layer by TF
tf.contrib.layers.fully_connected 
# since these layers are built in tensorflow, 
# we do have less control over what they do, so it's
# worth to use "scope": All variables we create within the scope will
# have the same prefix. This is helpful to mantain order. 

# Epsilon -> decrease linearly from 1 to 0.1 linearly, then remains
# 0.1 forever
