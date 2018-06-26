import tensorflow as tf
import numpy as np
import sys
import io

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# parameters
learning_rate = 1e-3
embed_size = 30
batch_size = 20
window_size = 13
hidden_size = 256
epochs = 1
keep_prob = 0.5

### HELPER FUNCTIONS ###
# preprocess training data
def preprocess_train(filename):

	# dictionary and indices for the dictionary
	vocab_dict = {}
	c = 0
	# read in the files
	with io.open(filename, mode = 'r', encoding = 'latin1') as f:
		# get lines
		lines = f.readlines()
		f.close()
	with io.open(filename, mode = 'r', encoding = 'latin1') as f:
		# get the words
		all_words = f.read()
		f.close()

	# get sentences
	sentences = [l.strip() for l in lines]
	# get words
	words = all_words.split()

	# get vocabulary
	vocab = set(words)
	for v in vocab:
		vocab_dict[v] = c
		c += 1

	# add "STOP" to the vocabulary
	stop_index = c
	vocab_dict["STOP"] = stop_index

	# padding the sentence to a length of 13
	# list of all sentences
	all_sentences = []
	# original lengths of the sentences (unpadded)
	length_list = []

	for sentence in sentences:
		sentence = sentence.split()
		# convert the sentence into integers
		sentence = [vocab_dict[word] for word in sentence]
		# store the O.G. length
		length_list.append(len(sentence)+1)
		# pad
		while (len(sentence)) < 13:
			sentence.append(stop_index)
		# add padded sentence to the list
		all_sentences.append(sentence)

	# convert to a numpy array
	all_sentences_np = np.asarray(all_sentences)
	return all_sentences_np, vocab_dict, length_list

# preprocess testing data
def preprocess_dev(filename, vocab_dict):

	# read in the lines
	with io.open(filename, mode = 'r', encoding = 'latin1') as f:
		lines = f.readlines()

	# get sentences
	sentences = [l.strip() for l in lines]
	# stop index in the dictionary
	stop_index = len(vocab_dict) - 1
	# list of all sentences
	all_sentences = []
	# list of sentence lengths (original, unpadded)
	length_list = []
	# list of amount of padding added to each sentence
	padding_list = []

	# convert sentences into word IDs, pad them, and store the unpadded sentences lengths, amount of padding added, and the sentences themselves
	for sentence in sentences:
		sentence = sentence.split()
		sentence = [vocab_dict[word] for word in sentence]
		# store both the length and the padding
		length_list.append(len(sentence) + 1)
		padding_list.append(12-len(sentence))
		# pad
		while (len(sentence) < 13):
			sentence.append(stop_index)
		# add to all sentences
		all_sentences.append(sentence)

	# convert to a numpy arrray
	all_sentences_np = np.asarray(all_sentences)
	return all_sentences_np, length_list, padding_list

# training step
def train(french_train_sentences, english_train_sentences, english_train_sentences_input, french_train_lengths, english_train_lengths):
	# initial state
	previousState = np.zeros([batch_size,hidden_size])
	# end condition
	fin = len(french_train_sentences)/batch_size * batch_size
	# iterate through the batches
	for step in range(0, fin, batch_size):
		# progress update
		print(str(step) +" / " + str(len(french_train_sentences)))
		# inputs, answers, sequences
		french_input = french_train_sentences[step: step + batch_size]
		english_input = english_train_sentences_input[step: step + batch_size]
		english_answer = english_train_sentences[step: step + batch_size]
		french_sequence = french_train_lengths[step: step + batch_size]
		english_sequence = english_train_lengths[step: step + batch_size]
		# dictionary and arguments
		feedDict = {encIn: french_input, decIn: english_input, answr: english_answer, initState: previousState, sequence: french_sequence, sequence2: english_sequence}
		sessArgs = [encState, perplexity, trainOp]
		# outputs
		hiddenStates, p, _ = sess.run(sessArgs, feedDict)
		# perplexity
		print p
		# update states
		previousState = hiddenStates

# testing step
def test(french_test_sentences, english_test_sentences, english_test_sentences_input, french_test_lengths, english_test_lengths):
	# initial state
	previousState = np.zeros([batch_size,hidden_size])
	# accuracy
	totalAccuracy = 0
	# counter
	count = 0
	# end condition
	fin = len(french_test_sentences)/batch_size * batch_size
	# global variables
	global accuracy
	global correctValues
	for step in range(0, fin, batch_size):
		# progress update
		print(str(step) +" / " + str(len(french_test_sentences)))
		# inputs, answers, sequences
		french_input = french_test_sentences[step: step + batch_size]
		english_input = english_test_sentences_input[step: step + batch_size]
		english_answer = english_test_sentences[step: step + batch_size]
		french_sequence = french_test_lengths[step: step + batch_size]
		english_sequence = english_test_lengths[step: step + batch_size]
		a = 0
		# dictionary and arguments
		feedDict = {encIn: french_input, decIn: english_input, answr: english_answer, initState: previousState, sequence: french_sequence, sequence2: english_sequence}
		sessArgs = [encState, correctValues]
		hiddenStates, c = sess.run(sessArgs, feedDict)

		# compute accuracy for the correct lengths - do not want to compute beyond the first stop
		for index, length in enumerate(english_sequence):
			# count the number of times the logits are correct
			realAccuracy = float(np.sum(c[index][:length]))/length
			# increment
			a += realAccuracy

		# compute accuracy
		a = a/batch_size
		totalAccuracy += a
		# print("Accuracy: " + str(totalAccuracy))
		previousState = hiddenStates
		count += 1

	# average accuracy
	print(str(totalAccuracy/count))

### ###

# parameters
french_train = sys.argv[1]
english_train = sys.argv[2]
french_test = sys.argv[3]
english_test = sys.argv[4]

# training data
french_train_sentences, french_vocab, french_train_lengths = preprocess_train(french_train)
english_train_sentences, english_vocab, english_train_lengths = preprocess_train(english_train)

# testing data
french_test_sentences, french_test_lengths, french_test_padding = preprocess_dev(french_test, french_vocab)
english_test_sentences, english_test_lengths, english_test_padding = preprocess_dev(english_test, english_vocab)

# vocab sizes
vfSz = len(french_vocab)
veSz = len(english_vocab)

# I FOUND THE FOLLOWING SYNTAX ONLINE, ON A SEQ2SEQ TUTORIAL
# all it does is add a stop to the start of the English sentences, by creating a column of stops and "stacking" it next to the English sentences
# we discard the last character of the English sentence, so as to maintain a length of 13

# formatting training input - veSz is the integer value for STOP
p = np.full(len(french_train_sentences), veSz-1)
english_train_sentences_input = np.column_stack((p, english_train_sentences[:, 0:12]))

p2 = np.full(len(french_test_sentences), veSz-1)
english_test_sentences_input = np.column_stack((p2, english_test_sentences[:, 0:12]))

### SANITY CHECK ###
# print len(french_train_sentences), len(english_train_sentences), len(french_train_padding), len(english_train_padding)
# print len(french_test_sentences), len(english_test_sentences), len(french_test_padding), len(english_test_padding)
# print french_train_sentences[1274], english_train_sentences[1274]
# print len(french_vocab), len(english_vocab)

# checking the sentence lengths
# for sentence in french_train_sentences:
# 	assert len(sentence) == 13
# for sentence in french_test_sentences:
# 	assert len(sentence) == 13
# for sentence in english_train_sentences:
# 	assert len(sentence) == 13
# for sentence in english_test_sentences:
# 	assert len(sentence) == 13
### ###

### TF Setup - the same as the textbook ###
# placeholders
encIn = tf.placeholder(tf.int32, shape=[batch_size, window_size])
decIn = tf.placeholder(tf.int32, shape=[batch_size, window_size])
answr = tf.placeholder(tf.int64, shape=[batch_size, window_size])
sequence = tf.placeholder(tf.float32, [batch_size])
sequence2 = tf.placeholder(tf.float32, [batch_size])

# encoder
with tf.variable_scope("enc"):
	F = tf.Variable(tf.random_normal((vfSz, embed_size), stddev=0.1))
	embs = tf.nn.embedding_lookup(F, encIn)
	embs = tf.nn.dropout(embs, keep_prob)
	cell = tf.contrib.rnn.GRUCell(hidden_size)
	initState = cell.zero_state(batch_size, tf.float32)
	encOut, encState = tf.nn.dynamic_rnn(cell, embs, initial_state=initState, sequence_length=sequence)

# decoder
with tf.variable_scope("dec"):
	E = tf.Variable(tf.random_normal((veSz, embed_size), stddev=0.1))
	embs = tf.nn.embedding_lookup(E, decIn)
	embs = tf.nn.dropout(embs, keep_prob)
	cell = tf.contrib.rnn.GRUCell(hidden_size)
	decOut,_ = tf.nn.dynamic_rnn(cell, embs, initial_state=encState, sequence_length=sequence2)

# weights and bias
W = tf.Variable(tf.random_normal([hidden_size, vfSz], stddev=0.1))
b = tf.Variable(tf.random_normal([vfSz], stddev=0.1))

# logits
logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b

# masking
m= tf.sequence_mask(sequence2, window_size, tf.float32)

# computing loss
loss = tf.contrib.seq2seq.sequence_loss(logits, answr, m)

# perplexity
perplexity = tf.exp(loss)

# I found this idea on the TF Seq2Seq Tutorial and Discussion
# list storing the number of correct logit values - 0 if wrong, 1 if right
correctValues = tf.cast(tf.equal(tf.argmax(logits,2), answr), tf.int32)

# accuracy
accuracy = tf.reduce_mean(tf.cast(correctValues, tf.float32))

# training step
trainOp = tf.train.AdamOptimizer(1e-3).minimize(loss)

### ###

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print("Initialization Complete.")

# print("TRAINING.")

train(french_train_sentences, english_train_sentences, english_train_sentences_input, french_train_lengths, english_train_lengths)

# print ("TESTING.")

test(french_test_sentences, english_test_sentences, english_test_sentences_input, french_test_lengths, english_test_lengths)
