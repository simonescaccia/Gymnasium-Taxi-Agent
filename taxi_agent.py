import gymnasium as gym
import numpy as np


class TaxiAgent:
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        env: gym.Env,
        discount_factor: float = 0.95
    ):
        """
        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        state_size = env.observation_space.n  # total number of states (S)
        action_size = env.action_space.n      # total number of actions (A)

        # Init Q-table with all zeros
        self.q_table = np.zeros(state_size, action_size)

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()       # random action

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_table[obs]))    # best action

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        # decrease epsilon: prefer exploration first, then exploitation 
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)