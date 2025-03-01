import numpy as np
import time

import gymnasium as gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from taxi_agent_DQN import TaxiAgentDQN
# hyperparameters
REPLAY_BUFFER_SIZE = 500 # Max batch size of past experience
BATCH_SIZE = 64 # Training set size
N_EPISODES = 5000
START_EPSILON = 1
EPSILON_DECAY = START_EPSILON / (N_EPISODES)  # reduce the exploration over time
FINAL_EPSILON = 0.0
NEURONS_FIRST_LAYER = 20
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.01

# SETTINGS
RENDER = False
 
if __name__ == "__main__":
    if RENDER:
        env = gym.make('Taxi-v3', render_mode="human")
    else:
        env = gym.make('Taxi-v3')

    agent = TaxiAgentDQN(
        initial_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        discount_factor=DISCOUNT_FACTOR,
        memory_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        neurons_first_layer=NEURONS_FIRST_LAYER,
        input_dimention=1,
        num_actions=env.action_space.n,
        learning_rate=LEARNING_RATE
    )

    # Train
    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=121)
        done = False
        episode_reward = 0
        episode_steps = 0
            
        # Start timer
        start = time.time()

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _  = env.step(action)
            episode_reward += reward
            episode_steps += 1

            agent.remember(obs, action, reward, next_obs, int(terminated))
            agent.decay_epsilon()
            agent.learn()
            
            done = terminated or truncated
            obs = next_obs

        print("Episode number: ", episode, 
              "Episode reward: ", episode_reward, 
              "Episode steps: ", episode_steps, 
              "Epsilon: ", agent.epsilon,
              "Episode time: ", time.time() - start)

