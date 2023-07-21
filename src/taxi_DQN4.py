from collections import deque
import random
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import gymnasium as gym
import time

from taxi_agent_DQN import TaxiAgentDQN


# hyperparameters
experience_max_size = 10000 # Max batch size of past experience
batch_size = 100 # Training set size
experience = deque([], experience_max_size) # Past experience arranged as a queue
start_epsilon = 1
epsilon_divider = 5000 
final_epsilon = 0.01
discount_factor = 0.99
neurons_first_layer = 64
neurons_second_layer = 32
scale_range_x = (-0.5, 0.5)

# SETTINGS
RENDER = False
LOAD_CHECKPOINT = False
SARSA_TRAINING = False
FILE_POSTFIX = "_sarsa" if SARSA_TRAINING else ""
CHECKPOINT_PATH = "nn_weights/weights{}_{}_{}/cp.ckpt".format(FILE_POSTFIX, neurons_first_layer, neurons_second_layer)

def choose_action(obs):
    # choose an action
    if np.random.random() < epsilon:
        # with probability epsilon return a random action to explore the environment
        action = env.action_space.sample()       # random action
    else:
        # with probability (1 - epsilon) act greedily (exploit)
        x_test = np.array([obs]).reshape(-1, 1)
        action = int(np.argmax(model.predict(x_test, verbose=0)))    # best action
    return action

def train_model(model: Sequential, experience):
    # train NN if experience contains at least BATCH_SIZE records
    if len(experience) >= batch_size:
        # sample batch
        batch = random.sample(experience, batch_size)
        # prepare data
        dataset = np.array(batch)
        
        # predict Q(s,a) given the batch of states
        next_reward = model.predict(dataset[:,3], verbose=0)
        x_train = dataset[:,0]
        y_train = np.empty((0, 6), float)
        for i in range(len(x_train)):
            entry = np.array([])
            for a in range(6):
                # append to entry
                entry = np.append(entry, dataset[i,2] + discount_factor * next_reward[0][a])
            y_train = np.append(y_train, [entry], axis=0)
        # train network
        model.fit(x_train, y_train, verbose=0) # fit model
        #model.save_weights(CHECKPOINT_PATH) # save updated weights

        # decrease epsilon: prefer exploration first, then exploitation
        global epsilon
        epsilon = max(final_epsilon, epsilon - epsilon/epsilon_divider)
 
if __name__ == "__main__":
    if RENDER:
        env = gym.make('Taxi-v3', render_mode="human")
    else:
        env = gym.make('Taxi-v3')

    taxi_agent = TaxiAgentDQN(
        initial_epsilon=start_epsilon,
        epsilon_divider=epsilon_divider,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        memory_size=experience_max_size,
        batch_size=batch_size,
        neurons_first_layer=neurons_first_layer,
        neurons_second_layer=neurons_second_layer,
        input_dimention=1,
        num_actions=env.action_space.n,
    )

    if LOAD_CHECKPOINT:
        # load checkpoint
        taxi_agent.load_model(CHECKPOINT_PATH)

    episode_reward = 0
    episode_number = 1
    # setup the environment
    obs, _ = env.reset()
    global epsilon
    epsilon = start_epsilon

    while(True):
        action = choose_action(obs)
        # execute the action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        # Record experience (will be used to train network)
        if len(experience)>=experience_max_size:
            experience.popleft() # dequeue oldest item
        # store the experience
        experience.append(np.array([obs, action, reward, next_obs])) 
        # train the model
        train_model(model, experience)
        obs = next_obs
        if terminated or truncated:
            print("Episode: {}, Reward: {}, experience: {}, epsilon: {}".format(
                episode_number, episode_reward, len(experience), epsilon))
            # reset the environment
            episode_number += 1
            obs, _ = env.reset()
            episode_reward = 0