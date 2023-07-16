from collections import defaultdict
import gymnasium as gym
import numpy as np


class TaxiAgent:
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        env: gym.Env,
    ):
        """
        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        action_size = env.action_space.n      # total number of actions (A)
        state_size = env.observation_space.n  # total number of states (S)

        # Init Q-table with all zeros
        self.q_table = np.zeros((state_size,action_size))

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.env = env

    def get_action(self, state: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()       # random action

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_table[state]))    # best action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        new_state: int
    ):
        """Updates the Q-value of an action."""

        # Qlearning algorithm: Q(s,a) := reward + discount_factor * max Q(s',a')
        self.q_table[state, action] = reward + self.discount_factor * np.max(self.q_table[new_state])

    def decay_epsilon(self):
        # decrease epsilon: prefer exploration first, then exploitation 
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
