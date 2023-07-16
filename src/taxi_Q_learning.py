# Import and create the gymnasium environment
import gymnasium as gym
from matplotlib.patches import Patch

from taxi_agent import TaxiAgent
from tqdm import tqdm # for progress bar
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# SETTINGS
DEBUG = False
TRAIN_AGENT = True
SAVE_TRAINING = False
PLOT_STATS = True
PLOT_POLICY = True
SHOW_AGENT_IN_ACTION = True
REPEAT_AGENT_IN_ACTION = 5

# hyperparameters
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
discount_factor = 0.95

def train_agent(env, agent):
    # Train the agent
    print("Training the agent...")
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # update the agent
            agent.update(state=obs, action=action, reward=reward, new_state=next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    if SAVE_TRAINING:
        # Save the q_table
        np.save("q_table_{}.npy".format(n_episodes), agent.q_table)

def plot_stats():
    # Plot the training statistics
    # A rolling average, sometimes referred to as a moving average, 
    # is a metric that calculates trends over short periods of time using a set of data. Specifically, 
    # it helps calculate trends when they might otherwise be difficult to detect.
    # x axis: rolling length
    rolling_length = 100
    _, axs = plt.subplots(ncols=2, figsize=(12, 5))
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

def plot_policy(agent):
    """Create policy grid given an agent."""
    # build a policy dictionary that maps observations to actions
    state = [0] * 500
    policy = [0] * 500
    for obs, action_values in enumerate(agent.q_table):
        state[obs] = int(np.max(action_values)) # get the max action value
        policy[obs] = int(np.argmax(action_values)) # get the index of the max action value

    state_count, y_count  = np.meshgrid(
        np.arange(0, 500),
        np.arange(0, 1)
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[obs[0]],
        axis=2,
        arr=np.dstack([state_count, y_count]),
    )

    """Creates a plot using the policy grid."""
    # plot the policy grid
    sns.set()
    # Show ticks, 0 to 499 without skipping for the x axis and 0 for the y axis
    ax = sns.heatmap(policy_grid, linewidth=0.5, annot=True, cmap="Accent_r", cbar=False)
    ax.set_title(f"Policy: Taxy policy after {n_episodes} episodes")
    ax.set_xlabel("State encoding")
    ax.set_ylabel("Y position")
    ax.set_xticks(range(0, 500, 5))
    ax.set_yticks([0])
    ax.set_xticklabels(range(0, 500, 5), rotation=0)
    ax.set_yticklabels([0], rotation=0)

    # create the legend
    legend_elements = [
        Patch(facecolor="grey", label="0: South"),
        Patch(facecolor="red", label="1: North"),
        Patch(facecolor="blue", label="2: East"),
        Patch(facecolor="yellow", label="3: West"),
        Patch(facecolor="pink", label="4: Pickup"),
        Patch(facecolor="green", label="5: Dropoff"),
    ]
    ax.legend(handles=legend_elements, loc = 'upper right')
    plt.show()

def show_agent_in_action(env, agent):
    # Watch trained agent
    print("Watching the agent...")
    for _ in range(REPEAT_AGENT_IN_ACTION):
        obs, _ = env.reset()
        done = False
        rewards = 0

        for s in range(n_episodes):
            action = np.argmax(agent.q_table[obs])
            if DEBUG: 
                print("Step {}".format(s+1))
                print(f"action: {action}")
            new_obs, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            if DEBUG: 
                print(f"score: {rewards}")
            done = terminated or truncated
            obs = new_obs

            if done == True:
                break

    env.close()

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    agent = TaxiAgent(
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        env=env,
    )

    if TRAIN_AGENT:
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
        train_agent(env, agent)
    else:
        # Load the q_table
        todo = 0
    
    if PLOT_STATS:
        plot_stats()
    if PLOT_POLICY:
        plot_policy(agent)

    if SHOW_AGENT_IN_ACTION:
        env = gym.make('Taxi-v3', render_mode='human')
        show_agent_in_action(env, agent)
    
    
