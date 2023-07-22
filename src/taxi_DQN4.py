import numpy as np

import gymnasium as gym

from taxi_agent_DQN import TaxiAgentDQN

# hyperparameters
replay_buffer_size = 10000 # Max batch size of past experience
batch_size = 64 # Training set size
start_epsilon = 1
epsilon_divider = 0.995
final_epsilon = 0.01
discount_factor = 0.95
neurons_first_layer = 64
neurons_second_layer = 64

# SETTINGS
RENDER = False
LOAD_CHECKPOINT = True
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
        epsilon_decrement=epsilon_divider,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        memory_size=replay_buffer_size,
        batch_size=batch_size,
        neurons_first_layer=neurons_first_layer,
        neurons_second_layer=neurons_second_layer,
        input_dimention=1,
        num_actions=env.action_space.n,
    )

    if LOAD_CHECKPOINT:
        # load checkpoint
        taxi_agent.load_model(CHECKPOINT_PATH)


    scores = []
    epsilon_history = []
    episode = 0

    while(True):
        done = False
        score = 0
        obs, _ = env.reset()
        actions = 0

        while not done:
            action = taxi_agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _  = env.step(action)
            score += reward
            taxi_agent.remember(obs, action, reward, next_obs)
            obs = next_obs
            done = terminated or truncated
            actions += 1

        taxi_agent.learn()

        episode += 1
        scores.append(score)

        avg_score = np.mean(scores[max(0, episode-100):(episode+1)]) # average score of last 100 episodes
        print('episode: ', episode,
              '-- score: ', score,
              '-- average score: %.2f' % avg_score,
              '-- epsilon: %.2f' % taxi_agent.epsilon,
              '-- actions: ', actions)

        if episode % 10 == 0 and episode > 0:
            taxi_agent.save_model(CHECKPOINT_PATH)

