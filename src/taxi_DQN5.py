import numpy as np
import gymnasium as gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from taxi_agent_DQN2 import TaxiAgentDQN

# Settings
MAX_EPISODES = 5000
NUM_EPISODES = 0

# Hyperparameters
MAX_MEM_SIZE = 10000
MAX_EPISODE_MEM_SIZE = 200
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.995
EPSILON_MAX = 1.00
EPSILON_MIN = 0.001
EPSILON_DEC = 0.999999
BATCH_SIZE = 64
NEURONS_FIRST_LAYER = 256
NEURONS_SECOND_LAYER = 256
ENV_NAME = 'Taxi-v3'
CHECKPOINT_PATH = "nn_weights/weights{}_{}_{}/cp.ckpt".format(ENV_NAME, NEURONS_FIRST_LAYER, NEURONS_SECOND_LAYER)

env = gym.make('Taxi-v3')

taxi_agent = TaxiAgentDQN(
    initial_epsilon=EPSILON_MAX,
    epsilon_decrement=EPSILON_DEC,
    final_epsilon=EPSILON_MIN,
    discount_factor=DISCOUNT_FACTOR,
    memory_size=MAX_MEM_SIZE,
    episode_memory_size=MAX_EPISODE_MEM_SIZE,
    batch_size=BATCH_SIZE,
    neurons_first_layer=NEURONS_FIRST_LAYER,
    neurons_second_layer=NEURONS_SECOND_LAYER,
    input_dimention=1,
    num_actions=env.action_space.n,
    learning_rate=LEARNING_RATE
)

scores = []
epsilon_history = []

for i in range(MAX_EPISODES):

    done = False
    obs, _ = env.reset()
    score = 0
    actions = 0
    taxi_agent.init_temporary_remember()

    while not done:
        action = taxi_agent.choose_action(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action) # Execute action and collect corresponding info     
        score += reward  # Cumulative reward
        taxi_agent.temporary_store_transition(obs, action, reward, obs_next, terminated) # Store experience in replay buffer
        taxi_agent.decrease_epsilon(i)
        obs = obs_next # Update state
        actions += 1
        done = terminated or truncated

    if terminated or (truncated and np.random.random() < 0.1):
        taxi_agent.store_temporary_transitions()
        taxi_agent.learn() # Train the agent

    scores.append(score)

    avg_score = np.mean(scores[max(0, i-100):(i+1)]) # average score of last 100 episodes
    print('episode: ', i,
            '-- score: ', score,
            '-- average score: %.2f' % avg_score,
            '-- epsilon: %.2f' % taxi_agent.epsilon,
            '-- actions: ', actions)

    if i % 10 == 0 and i > 0:
        taxi_agent.save_model(CHECKPOINT_PATH)
