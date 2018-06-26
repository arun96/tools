import tensorflow as tf
import numpy as np
import sys

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# parameters
learning_rate = 1e-4
hidden = 128 #size of hidden layer
embed_size = 30
batch_size = 20
epochs = 1
keepP = 0.85

# Makes the weight variable of a given shape
def weight_variable(shape):
	# normal distribution, s.d. of 0.1
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Makes the bias variable of a given shape
def bias_variable(shape):
	# bias = 0.1
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Helper function to pre-process input data
def pre_process(train_path, dev_path):
	train_file = open(train_path, "rb")
	dev_file = open(dev_path, "rb")

	# File in terms of IDs
	train_list = []
	dev_list = []

	# Dictionary mapping words to IDs
	word_to_id = {}

	# Assigning IDs for train file
	v = 0
	for line in train_file:
		for word in line.split():
			if word not in word_to_id:
				word_to_id[word] = v
				v += 1 # Increment ID
			train_list.append(word_to_id[word])

	# Dev files
	for line in dev_file:
		for word in line.split():
			dev_list.append(word_to_id[word])

	return train_list, dev_list, word_to_id

# Main function
def main(train_path, dev_path):

	# start the session
	sess = tf.InteractiveSession()

	# get the training list, development list, word to id dictionary
	train_list, dev_list, word_to_id = pre_process(train_path, dev_path)

	# size of the vocabulary
	vocab_size = len(word_to_id)

	# reverse the dictionary too
	id_to_word = {v: k for k, v in word_to_id.iteritems()}

	# initialize as in the textbook
	inpt = tf.placeholder(tf.int32, shape=[None])
	inpt2 = tf.placeholder(tf.int32, shape=[None])
	answr = tf.placeholder(tf.int32, shape=[None])

	E = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev = 0.1))

	# embeddings
	embed = tf.nn.embedding_lookup(E, inpt)
	embed2 = tf.nn.embedding_lookup(E, inpt2)
	both = tf.concat([embed, embed2],1)

	# first layer
	w1 = weight_variable([2*embed_size, hidden])
	b1 = bias_variable([hidden])
	h = tf.nn.relu(tf.matmul(both, w1) + b1)

	# drop out - keep prob defined above
	keep_prob = tf.placeholder(tf.float32)
	h_drop = tf.nn.dropout(h, keep_prob)

	# second layer
	w2 = weight_variable([hidden, vocab_size])
	b2 = bias_variable([vocab_size])
	logits = tf.matmul(h_drop, w2) + b2

	# cross entropy loss
	xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=answr)

	# error & perplexity
	error = tf.reduce_mean(xEnt)
	perplexity = tf.exp(error)

	# training step
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

	# run session
	sess.run(tf.global_variables_initializer())

	# Training phase
	print("Training")

	for epoch in range(epochs):
		# Start with first word
		x = 0
		# Iterate through all words in file, one batch at a time
		while (x + batch_size + 2) < (len(train_list)):
			# Inputs = first two words of trigram, Answer = third word
			_, err = sess.run([train_step, error], feed_dict = {inpt: train_list[x : x+batch_size], inpt2: train_list[x+1 : x+1+batch_size], answr: train_list[x+2 : x+2+batch_size], keep_prob: keepP})

			# Print perplexity every 10000
			if x % 10000 == 0:
				print("perplexity for " + str(x) + ": %g"%perplexity.eval(feed_dict = {inpt: train_list[x : x+batch_size], inpt2: train_list[x+1 : x+1+batch_size], answr: train_list[x+2 : x+2+batch_size], keep_prob: keepP}))
			
			# Consider next batch
			x += batch_size

	# Testing phase
	print("Testing")

	# Pass in the development data
	err = sess.run([error], feed_dict = {inpt: dev_list[0 : len(dev_list)-2], inpt2: dev_list[1 : len(dev_list)-1], answr: dev_list[2 : len(dev_list)], keep_prob: 1.0})
	
	# Development data perplexity
	print("%g"%perplexity.eval(feed_dict = {inpt: dev_list[0 : len(dev_list)-2], inpt2: dev_list[1 : len(dev_list)-1], answr: dev_list[2 : len(dev_list)], keep_prob: 1.0}))


if __name__ == '__main__':
	# run.
	main(sys.argv[1], sys.argv[2])