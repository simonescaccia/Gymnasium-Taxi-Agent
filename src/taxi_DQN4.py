import numpy as np

import gymnasium as gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from taxi_agent_DQN import TaxiAgentDQN

# hyperparameters
replay_buffer_size = 5000 # Max batch size of past experience
temporary_replay_buffer_size = 200 # Max batch size of experience in current episode
batch_size = 200 # Training set size
start_epsilon = 1
epsilon_decrement = 0.9995
final_epsilon = 0.01
discount_factor = 0.99
neurons_first_layer = 50
neurons_second_layer = 50
learning_rate = 0.005

# SETTINGS
RENDER = False
LOAD_CHECKPOINT = False
SARSA_TRAINING = False
FILE_POSTFIX = "_sarsa" if SARSA_TRAINING else ""
CHECKPOINT_PATH = "nn_weights/weights{}_{}_{}/cp.ckpt".format(FILE_POSTFIX, neurons_first_layer, neurons_second_layer)
 
if __name__ == "__main__":
    if RENDER:
        env = gym.make('Taxi-v3', render_mode="human")
    else:
        env = gym.make('Taxi-v3')

    taxi_agent = TaxiAgentDQN(
        initial_epsilon=start_epsilon,
        epsilon_decrement=epsilon_decrement,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        memory_size=replay_buffer_size,
        temporary_memory_size=temporary_replay_buffer_size,
        batch_size=batch_size,
        neurons_first_layer=neurons_first_layer,
        neurons_second_layer=neurons_second_layer,
        input_dimention=1,
        num_actions=env.action_space.n,
        learning_rate=learning_rate
    )

    if LOAD_CHECKPOINT:
        # load checkpoint
        taxi_agent.load_model(CHECKPOINT_PATH)

    scores = []
    epsilon_history = []
    episode = 0

    # Train
    while(True):
        done = False
        score = 0
        obs, _ = env.reset()
        actions = 0

        while not done:
            action = taxi_agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _  = env.step(action)
            score += reward
            taxi_agent.remember(obs, action, reward, next_obs, int(terminated))
            obs = next_obs
            taxi_agent.learn()
            done = terminated or truncated
            actions += 1   

        taxi_agent.decrement_epsilon(1) 

        episode += 1
        scores.append(score)

        avg_score = np.mean(scores[max(0, episode-100):(episode+1)]) # average score of last 100 episodes
        print('episode: ', episode,
              '-- score: ', score,
              '-- average score: %.2f' % avg_score,
              '-- epsilon: %.2f' % taxi_agent.epsilon,
              '-- actions: ', actions)

        taxi_agent.save_model(CHECKPOINT_PATH)

