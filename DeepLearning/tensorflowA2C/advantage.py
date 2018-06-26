import tensorflow as tf
import gym
import numpy as np

# Supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Constants
learning_rate = 0.001
trials = 3
episodes = 1000
moves = 999
discount = 0.99
hidden_size = 32
critic_size = 128
updates = 50

# Reset the game
game = gym.make("CartPole-v1")
game.reset()

# Layer 1
state = tf.placeholder(shape=[None,4],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4,hidden_size],dtype=tf.float32))
bias = tf.Variable(tf.random_uniform([hidden_size], dtype=tf.float32))
hidden = tf.nn.relu(tf.matmul(state,W)+ bias)

# Layer 2
O = tf.Variable(tf.random_uniform([hidden_size,2],dtype=tf.float32))
bias2 = tf.Variable(tf.random_uniform([2], dtype=tf.float32))
output = tf.nn.softmax(tf.matmul(hidden,O) + bias2)

# Placeholders 
rewards = tf.placeholder(shape=[None], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
indices = tf.range(0, tf.shape(output)[0])*2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]), indices)
loss = -tf.reduce_mean(tf.log(actProbs) * rewards)

# Placeholders for critic loss
V1 = tf.Variable(tf.random_normal([4,critic_size],dtype=tf.float32,stddev=.1))
v1Out = tf.nn.relu(tf.matmul(state,V1))
V2 = tf.Variable(tf.random_normal([critic_size,1],dtype=tf.float32,stddev=.1))
vOut = tf.matmul(v1Out,V2)
vLoss = tf.reduce_mean(tf.square(rewards-vOut))
loss = loss + vLoss

optimizer = tf.train.AdamOptimizer(learning_rate)
trainOp = optimizer.minimize(loss)

avgRs = []

# Helper function to generate distribution
def generate_disRs(hist):
	dist = []
	last_reward = 0
	for element in reversed(hist):
		reward = discount * last_reward + element 
		dist.append(reward)
		last_reward = reward
	return list(reversed(dist))

# Training
for trial in range(trials):
	# Re-initialize session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# List to store total rewards
	totRs = []

	for episode in range(episodes):

		# reset
		st = game.reset()

		# List to store state, action and reward histories
		state_hist = []
		action_hist = []
		reward_hist = []

		# List to store history of states and values
		state_value_hist = []

		for move in range(moves):

			# Run
			actDict, stateVal = sess.run([output, vOut], feed_dict={state: [st]})

			# Get the random action
			action = np.random.choice(np.array([0,1]), p=actDict[0])

			st1, reward, done, info = game.step(action)

			# Render the game
			# game.render()

			# Add to the history
			action_hist.append(action)
			reward_hist.append(reward)
			state_hist.append(st)

			state_value_hist.append(stateVal[0][0])

			# Iterate
			st = st1

			# Update
			if done or (move%updates == 0 and move != 0):
				# Get disRs
				disRs = generate_disRs(reward_hist)

				# Compute Difference
				difference = np.array(disRs) - np.array(state_value_hist)

				# Run
				feed_dict = {state: state_hist, actions: action_hist, rewards: difference}
				l, _ = sess.run([loss, trainOp], feed_dict=feed_dict)

				if done:
					totRs.append(move)
					# print move, disRs[0]
					break


	# print and then append
	print str(np.average(totRs[900: 1000]))
	avgRs.append(np.average(totRs[900: 1000]))

# Final Reward across three trials
print avgRs
print(str(np.average(avgRs)))