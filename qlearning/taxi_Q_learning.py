import gymnasium as gym

from taxi_agent import TaxiAgent

N_EPISODES = 5000
START_EPSILON = 1.0
FINAL_EPSILON = 0.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES)  # reduce the exploration over time
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.01

if __name__ == "__main__":
    # Create the environment
    env = gym.make('Taxi-v3')
    agent = TaxiAgent(
        initial_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        env=env,
    )
    # Train the agent
    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            agent.update(state=obs, action=action, reward=reward, new_state=next_obs)

            agent.decay_epsilon()

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        print("Episode number: ", episode, "Episode reward: ", episode_reward, "Episode steps: ", episode_steps, "Epsilon: ", agent.epsilon)

        