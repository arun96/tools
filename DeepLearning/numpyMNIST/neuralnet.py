import os
import struct
import numpy as np
import gzip
import matplotlib.pyplot as plt
import copy
import sys

# Main function
def main(xtrain, ytrain, xtest, ytest):

	# Learning Rate
	L = np.float64(0.5)

	# Get the files
	X_train, y_train = load_mnist(xtrain, ytrain)
	#print('Training - Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

	X_test, y_test = load_mnist(xtest, ytest)
	#print('Testing - Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

	# Normalize
	X_test = X_test / 255.00
	X_train = X_train / 255.00

	# Get on hot representations
	y_train_onehot = one_hot(y_train)
	y_test_onehot = one_hot(y_test)

	# Initialize Weights + Bias
	w, b = initialize_weights()
	# w[i][j] = jth value in ith row = jth perceptron working on ith feature

	# The 10000 distinct images to train on
	selected = np.random.choice(60000, size = 10000, replace = False)
	# print selected

	# Iterate through the training set 
	for i in selected:

		# --- FORWARD PASS ---

		# L = XW + B
		val = np.dot(X_train[[i],:], w)
		logits = val + b

		# Softmax
		s = softmax(logits)

		# Get the correct probability, and compute Loss = -ln(p(a))
		idx = y_train[i]
		s_answer = s[0][idx]

		loss = -1.0 * np.log(s_answer)

		# --- BACKWARD PASS ---

		x_transpose = copy.deepcopy(X_train[[i],:])

		x_transpose = np.transpose(x_transpose)

		# Update Biases and Weights

		for p in range(0, len(b[0])):

			if (p == idx):

				b[0][p] = b[0][p] + (L * (1.0 - s_answer))

				w[:,[p]] = w[:,[p]] + x_transpose * -L * -(1.0 - s_answer)

			else:

				b[0][p] = b[0][p] + (L * (-1.0 * s[0][p]))

				w[:,[p]] = w[:,[p]] + (x_transpose * -L * (s[0][p]))

	test(X_test,y_test, w, b)


def test(X_test, y_test, w, b):

	c = 0

	for i in range(0, len(X_test)):

		logits = np.dot(X_test[[i],:], w) + b

		label = np.argmax(logits[0])

		if label == y_test[i]:

			c = c + 1

	print (float(c)/float(len(X_test)))

# Helper function to initialize weights and bias matrix
def initialize_weights():
	w = np.zeros((784,10))
	b = np.zeros((1, 10))
	return w, b

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

# Helper function for loading the mnist data
def load_mnist(images_path, labels_path):

	with gzip.open(labels_path, 'rb') as l:
		l.read(8)
		buffer = l.read()
		labels = np.frombuffer(buffer, dtype=np.uint8)

	with gzip.open(images_path, 'rb') as i:
		i.read(16)
		buffer = i.read()
		images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 784).astype(np.float64)
 
	return images, labels

# Converts the labels into one hot values
def one_hot(y):
	onehot = np.zeros((10, y.shape[0]))
	for idx, val in enumerate(y):
		onehot[val, idx] = 1.0
	return onehot

if __name__ == '__main__':

	# python neuralnet.py /Users/Arun/Desktop/Fall2017/CSCI1470/hw1/train-images-idx3-ubyte.gz /Users/Arun/Desktop/Fall2017/CSCI1470/hw1/train-labels-idx1-ubyte.gz /Users/Arun/Desktop/Fall2017/CSCI1470/hw1/t10k-images-idx3-ubyte.gz /Users/Arun/Desktop/Fall2017/CSCI1470/hw1/t10k-labels-idx1-ubyte.gz
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])