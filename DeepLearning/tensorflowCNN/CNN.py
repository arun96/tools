import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# THIS IMPLEMENTATION IS BASED HEAVILY ON THE DEEP MNIST TUTORIAL FROM THE TENSORFLOW WEBSITE

# function to build the graph for a deep net that classifies digits
# takes in an input tensor with dimensions (# of examples, 784)
# returns a tensor of shape (# examples , 10)
def deepnn(x):
	
	# first, we reshape
	# only one feature, as image is grayscale
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])

	# first convolutional layer will map one grayscale image to 32 feature maps
	with tf.name_scope('conv1'):
		w_c1 = weight_variable([5, 5, 1, 32])
		b_c1 = bias_variable([32])
		h_c1 = tf.nn.relu(conv2d(x_image, w_c1) + b_c1)

	# pooling - downsamples by 2X
	with tf.name_scope('pool1'):
		h_p1 = max_pool_2x2(h_c1)

	# second convolutional layer - we map 32 feature maps to 64
	with tf.name_scope('conv2'):
		w_c2 = weight_variable([5, 5, 32, 64])
		b_c2 = bias_variable([64])
		h_c2 = tf.nn.relu(conv2d(h_p1, w_c2) + b_c2)

	# second pooling layer - downsamples by 2X
	with tf.name_scope('pool2'):
		h_p2 = max_pool_2x2(h_c2)

	# we now have 7x7x64 feature maps
	# fully connected layer 1 - map these to 1024 features
	with tf.name_scope('fc1'):
		w_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])

		h_p2_flat = tf.reshape(h_p2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_p2_flat, w_fc1) + b_fc1)

	# next fully connected layer - converts the 1024 to 10 classes
	with tf.name_scope('fc2'):
		w_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])

		y_conv = tf.matmul(h_fc1, w_fc2) + b_fc2

	# finally, return tensor
	return y_conv

# helper functions - written using the tensorflow tutorial

# returns a 2d convolution layer
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max pooling to downsample a feature map by 2X
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# makes the weight variable of a given shape
def weight_variable(shape):
	# normal distribution, s.d. of 0.1
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# makes the bias variable of a given shape
def bias_variable(shape):
	# bias = 0.1
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# main function - runs the code, trains on the data, tests, and reports accuracy
def main():

	# variables for batches and learning rate
	num_batches = 2000
	batch_size = 50
	learning_rate = 1e-4

	# how often to print progress
	print_rate = 200

	# use the read_data_ssets function to get the Mnist data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# model
	x = tf.placeholder(tf.float32, [None, 784])

	# loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])

	# build the deep net graph
	y_conv = deepnn(x)

	# cross entropy loss
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
	# reduce cross entropy loss
	cross_entropy = tf.reduce_mean(cross_entropy)

	# using adam optimizer, with learning rate of 1E-4, minimizing the cross entropy loss
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	# accuracy
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
	# reduce
	accuracy = tf.reduce_mean(correct_prediction)

	# run the session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_batches):
			batch = mnist.train.next_batch(batch_size)

			# UNCOMMENT to every so often, print an update on accuracy
			# if i % print_rate == 0:
			# 	train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
			# 	print('step %d, training accuracy %g' % (i, train_accuracy))

			# train
			train_step.run(feed_dict={x: batch[0], y_: batch[1]})

		# print test accuracy
		print('%g' % accuracy.eval(feed_dict={
				x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
	# run.
	main()