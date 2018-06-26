import tensorflow as tf
import gym
import numpy as np
import io

# supresses compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Constants
learning_rate = 0.005
trials = 3
episodes = 1000
moves = 999
discount = 0.9999
hidden_size = 32

# Set up game
game = gym.make("CartPole-v0")
game.reset()

# TF Graph Instructions
state = tf.placeholder(shape=[None,4],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4,hidden_size],dtype=tf.float32))
hidden = tf.nn.relu(tf.matmul(state,W))
O = tf.Variable(tf.random_uniform([hidden_size,2],dtype=tf.float32))
output = tf.nn.softmax(tf.matmul(hidden,O))

rewards = tf.placeholder(shape=[None], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
indices = tf.range(0, tf.shape(output)[0])*2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]), indices)
loss = -tf.reduce_mean(tf.log(actProbs) * rewards)
optimizer = tf.train.AdamOptimizer(learning_rate)
trainOp = optimizer.minimize(loss)

# List of average rewards
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


# Policy Gradient Training
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

		for move in range(moves):

			# Run
			actDist = sess.run(output, feed_dict={state:[st]})

			# Get the random action
			action = np.random.choice(np.array([0,1]), p=actDist[0])

			# Get the state, reward, and done check
			st1, reward, done, _ = game.step(action)

			# Render the game
			# game.render()

			# Add to the history
			action_hist.append(action)
			reward_hist.append(reward)
			state_hist.append(st)

			# Iterate
			st = st1

			# If Done
			if done:
				# Get disRs
				disRs = generate_disRs(reward_hist)
				feed_dict = {state: state_hist, actions: action_hist, rewards: disRs}
				# Train
				l,_ = sess.run([loss,trainOp], feed_dict=feed_dict)

				# Append
				totRs.append(disRs[0])

				break

		# Print updates
		# if episode%100 == 0 and episode!= 0:
			# print(np.average(totRs[episode-100: episode])) 

	# Average of last 100 episodes
	avgRs.append(np.average(totRs[900: 1000]))
	# print(str(np.average(totRs[900: 1000])))

# Final Reward across three trials
print avgRs
print(str(np.average(avgRs)))