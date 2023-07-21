from collections import deque
import random
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import gymnasium as gym


# hyperparameters
experience_max_size = 10000 # Max batch size of past experience
batch_size = experience_max_size // 10 # Training set size
experience = deque([], experience_max_size) # Past experience arranged as a queue
start_epsilon = 0.7
final_epsilon = 0.3
discount_factor = 0.95
episode_reward = 0
episode_check = 1
neuron_first_layer = 64
neuron_second_layer = 64
scale_range_x = (-0.5, 0.5)

# SETTINGS
RENDER = False
LOAD_CHECKPOINT = False
SARSA_TRAINING = False
FILE_POSTFIX = "_sarsa" if SARSA_TRAINING else ""
CHECKPOINT_PATH = "nn_weights/weights{}_{}_{}/cp.ckpt".format(FILE_POSTFIX, neuron_first_layer, neuron_second_layer)

# NN architecture: (s, a) -> Q(s, a) 
model = Sequential()
model.add(Dense(neuron_first_layer, input_shape=(2,), activation='relu'))
model.add(Dense(neuron_second_layer, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='sgd', loss='mse')

if LOAD_CHECKPOINT:
    # load checkpoint
    model.load_weights(CHECKPOINT_PATH)

if RENDER:
    env = gym.make('Taxi-v3', render_mode="human")
else:
    env = gym.make('Taxi-v3')

episode_number = 1
# setup the environment
obs, _ = env.reset()
epsilon = start_epsilon
# scale the data between -0.5 and 0.5 to avoid exploding gradients: x belongs to [0, 499], y belongs to [-10, 20]
scale_x = MinMaxScaler(feature_range=scale_range_x)
scale_x = scale_x.fit(np.array([[0, 0], [499, 5]]))

while(True):
    # choose an action
    if np.random.random() < epsilon:
        # with probability epsilon return a random action to explore the environment
        action = env.action_space.sample()       # random action
    else:
        # with probability (1 - epsilon) act greedily (exploit)
        q_values = []
        for act in range(6):
            # get the q_value from the nn
            x_test = np.array([obs, act]).reshape(1, 2)
            # scale the data
            x_test = scale_x.transform(x_test)
            q_values.append(model.predict(x_test, verbose=0))
        action = int(np.argmax(q_values))    # best action
    # execute the action
    next_obs, reward, terminated, truncated, _ = env.step(action)

    episode_reward += reward

    # store the experience, use deterministic Q-learning
    # compute the best action for the next state
    q_values = []
    for act in range(6):
        # get the q_value from the nn
        x_test = np.array([next_obs, act]).reshape(1, 2)
        # scale the data
        x_test = scale_x.transform(x_test)
        q_values.append(model.predict(x_test, verbose=0))
    next_reward = float(np.max(q_values))    # best reward

	# Record experience (will be used to train network)
    if len(experience)>=experience_max_size:
        experience.popleft() # dequeue oldest item

    # store the experience
    experience.append([obs, action, reward + discount_factor * next_reward])

    obs = next_obs

    if terminated or truncated:
        # train NN if experience contains at least BATCH_SIZE records
        if len(experience) >= batch_size:
			# sample batch
            batch = random.sample(experience, batch_size)
            # prepare data
            dataset = np.array(batch)
            x_train = dataset[:,:2]
            y_train = dataset[:,1]
            # scale the data
            x_train = scale_x.transform(x_train)
            # train network
            model.fit(x_train, y_train) # fit model
            model.save_weights(CHECKPOINT_PATH) # save updated weights

            # decrease epsilon: prefer exploration first, then exploitation 
            epsilon = max(final_epsilon, epsilon - epsilon/100)

        print("Episode: {}, Reward: {}, experience: {}, episode_number: {}, epsilon: {}".format(
            episode_number, episode_reward, len(experience), episode_number, epsilon))
        # reset the environment
        episode_number += 1
        obs, _ = env.reset()
        episode_reward = 0
 