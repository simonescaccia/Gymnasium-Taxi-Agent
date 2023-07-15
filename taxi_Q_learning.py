# Import and create the gymnasium environment
import gymnasium as gym
env = gym.make('Taxi-v3')

from taxi_agent import TaxiAgent
from tqdm import tqdm # for progress bar
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict # for dictionary with default value
import seaborn as sns

# hyperparameters
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
discount_factor = 0.95

agent = TaxiAgent(
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
    env=env,
)

# Train the agent
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(state=obs, action=action, reward=reward, new_state=next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

# Plot the training statistics
rolling_length = 500
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

plt.tight_layout()
plt.show()

# Visualizing the policy
def create_grids(agent):
    """Create value and policy grid given an agent."""
    # build a policy dictionary that maps observations to actions
    state_value = defaultdict(int)
    policy = defaultdict(int)
    for obs, action_values in agent.q_table.items():
        state_value[obs] = int(np.max(action_values)) # get the max action value
        policy[obs] = int(np.argmax(action_values)) # get the index of the max action value

    y_count, state_count = np.meshgrid(
        np.arange(0, 1),
        np.arange(0, 500),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1])],
        axis=2,
        arr=np.dstack([state_count, y_count]),
    )
    return policy_grid


def create_plots(policy_grid, title: str):
    """Creates a plot using the policy grid."""
    # plot the policy
    print("Policy grid: ", policy_grid)
    sns.set()
    ax = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax.set_title(f"Policy: {title}")
    ax.set_xlabel("State encoding")
    ax.set_xticklabels(range(0, 500))
    ax.set_ylabel("Y position")
    ax.set_yticklabels(range(0, 1))
    plt.show()

policy_grid = create_grids(agent)
create_plots(policy_grid, "Taxi-v3 policy")
