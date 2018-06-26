import tensorflow as tf
import numpy as np
import sys

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# parameters
learning_rate = 1e-3
embed_size = 100
batch_size = 50
window_size = 20
rnn_size = 256
epochs = 1

# helper function to preprocess the training data, and generate window and batches, and the lists and dictionaries
def preprocessing_train(filename):
	train_file = open(filename, "rb")

	# list of words
	train_words = []

	# file in terms of IDs
	train_list = []

	# dictionary mapping words to IDs
	word_to_id = {}
	
	# assigning IDs for the training file
	v = 0
	for line in train_file:
		for word in line.split():
			if word not in word_to_id:
				word_to_id[word] = v
				v += 1
			train_words.append(word)
			train_list.append(word_to_id[word])

	# set up the windows
	window_data = []
	window_labels = []

	for i in xrange(0, len(train_words) - window_size - 1, window_size):
		data = train_list[i:i+window_size]
		labels = train_list[i+1:i+window_size+1]
		window_data.append(data)
		window_labels.append(labels)

	# set up the batches
	batches_data = []
	batches_labels = []
	num_batch = len(window_data)/batch_size

	for batch_idx in xrange(0, num_batch):
		batch_data = [window_data[i * num_batch+batch_idx] for i in xrange(batch_size)]
		batch_label = [window_labels[i*num_batch+batch_idx] for i in xrange(batch_size)]
		batches_data.append(batch_data)
		batches_labels.append(batch_label)

	return train_words, train_list, word_to_id, batches_data, batches_labels, num_batch

# helper function to preprocess the dev data, and generate window and batches, and the lists and dictionaries
def preprocessing_dev(filename, word_to_id):

	dev_file = open(filename, "rb")

	#list of all words
	dev_words = []

	# file in terms of IDs
	dev_list = []

	# fill in dev_list
	for line in dev_file:
		for word in line.split():
			dev_list.append(word_to_id[word])
			dev_words.append(word)

	# set up windows
	window_data = []
	window_labels = []

	for i in xrange(0, len(dev_words) - window_size - 1, window_size):
		data = dev_list[i:i+window_size]
		labels = dev_list[i+1:i+window_size+1]
		window_data.append(data)
		window_labels.append(labels)

	# set up batches
	batches_data = []
	batches_labels = []
	num_batch = len(window_data)/batch_size

	for batch_index in xrange(0, num_batch):
		batch_data = [window_data[i*num_batch+batch_index] for i in xrange(batch_size)]
		batch_labels = [window_labels[i*num_batch+batch_index] for i in xrange(batch_size)]
		batches_data.append(batch_data)
		batches_labels.append(batch_labels)

	return dev_words, dev_list, batches_data, batches_labels, num_batch

# makes the weight variable of a given shape
def weight_variable(shape):
	# normal distribution, s.d. of 0.1
	initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# makes the bias variable of a given shape
def bias_variable(shape):
	# bias = 0.1
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# helper function to run the training stage
def train(batches_data, batches_labels, num_batch):
	previousState = np.zeros([batch_size, rnn_size])
	for batch in xrange(num_batch):

		batch_data = batches_data[batch]
		batch_label = batches_labels[batch]

		feedDict = {inpt: batch_data, output: batch_label, initialState: previousState}
		sessArgs = [nextState, perplexity, train_step]
		hiddenStates, p, _ = sess.run(sessArgs, feedDict)
		previousState = hiddenStates
		# print (str(batch) + "/" + str(num_batch)), (p)

# helper function to run the development stage
def dev(dev_batches_data, dev_batches_labels, dev_num_batches):
	previousState = np.zeros([batch_size, rnn_size])
	total_p = 0

	for batch in xrange(dev_num_batches):
		batch_data = dev_batches_data[batch]
		batch_label = dev_batches_labels[batch]

		feedDict = {inpt: batch_data, output: batch_label, initialState: previousState}
		sessArgs = [nextState, perplexity]
		hiddenStates, p = sess.run(sessArgs, feedDict)
		previousState = hiddenStates
		total_p += p

	print total_p/dev_num_batches

# get the two files
train_path = sys.argv[1]
dev_path = sys.argv[2]

### PREPROCESSING #####

# get the training words, list, the word to ID dictionary, batches of data and labels, and the number of batches
train_words, train_list, word_to_id, batches_data, batches_labels, num_batches = preprocessing_train(train_path)
vocab_size = len(word_to_id)

# get the dev words, list, batches of data and labels, and the number of batches
dev_words, dev_list, dev_batches_data, dev_batches_labels, dev_num_batches = preprocessing_dev(dev_path, word_to_id)

##### SET UP THE RNN #####

# initialize
inpt = tf.placeholder(tf.int32, [batch_size, window_size])
output = tf.placeholder(tf.int32, [batch_size, window_size])

# embedding
E = tf.Variable(tf.random_normal([vocab_size, embed_size], stddev = 0.1))
embed = tf.nn.embedding_lookup(E, inpt)

# rnn
rnn= tf.contrib.rnn.GRUCell(rnn_size) # adds RNN to the computation graph
initialState = rnn.zero_state(batch_size, tf.float32)
rnn_output, nextState = tf.nn.dynamic_rnn(rnn, embed, initial_state=initialState)

# reshape the output
rnn_output = tf.reshape(rnn_output, [batch_size*window_size, rnn_size])

# weights and bias variables - TODO: check bias
W = weight_variable([rnn_size, vocab_size])
b = bias_variable([vocab_size])

# logits
logits = tf.matmul(rnn_output, W) + b

# reshape
logits = tf.reshape(logits,[batch_size, window_size, vocab_size])

# new weights - just have to be ones
weights = tf.ones([batch_size, window_size])

# loss
loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=output, weights=weights)

# perplexity
perplexity = tf.exp(loss)

# training step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# start and run the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

##### TRAINING AND TESTING #####

# training step
# print "TRAINING"

train(batches_data, batches_labels, num_batches)

# testing step
# print "TESTING"

dev(dev_batches_data, dev_batches_labels, dev_num_batches)