import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size: int):
        self.mem_counter = 0    # count the number of entry in the buffer
        self.mem_size = max_size    # maximum number of entry in the buffer
        self.state_memory = np.zeros(self.mem_size)  # contains the state
        self.new_state_memory = np.zeros(self.mem_size) # contains the new state
        self.action_memory = np.zeros(self.mem_size, dtype=int)    # contains the action
        self.reward_memory = np.zeros(self.mem_size)    # contains the reward

    def store_transition(self, state, action, reward, new_state):
        index = self.mem_counter % self.mem_size    # replace the oldest entry
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)   # sample batch_size entries from the buffer

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]

        return states, actions, rewards, new_states