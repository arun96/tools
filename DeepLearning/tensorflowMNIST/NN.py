import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# BASED ON THE SOFTMAX TUTORIAL ON THE TENSORFLOW SITE

def main():

	batch_size = 100
	learning_rate = 0.5

	# Import data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])

	# Weights and biases of the two layers - using 256 as the size of hidden layer
	W = tf.Variable(tf.random_normal([784, 256], stddev=0.1)) 
	W1 = tf.Variable(tf.random_normal([256, 10], stddev=0.1)) # softmax layer is of size 10
	b = tf.Variable(tf.zeros([256])) # 256 bias values for hidden
	b1 = tf.Variable(tf.zeros([10])) # 10 bias values of softmax
	y = tf.matmul(x, W) + b # outputs for hidden layer
	y = tf.nn.relu(y) # rectified unit layer - f(x) = max(0, f(x))
	y = tf.matmul(y, W1) + b1 # softmax layer logits

	# Define loss and optimizer - just like the tutorial
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

	# From tutorial - start session and initialize
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Train - we will do 2000 iterations
	for _ in range(2000):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size) # batch size = 100
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Print the accuracy of testing
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':

	main()