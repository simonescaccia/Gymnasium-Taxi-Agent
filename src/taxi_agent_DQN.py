import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from replay_buffer import ReplayBuffer


class TaxiAgentDQN:
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_divider: float,
        final_epsilon: float,
        discount_factor: float,
        memory_size: int,
        batch_size: int,
        neurons_first_layer: int,
        neurons_second_layer: int,
        input_dimention: int,
        num_actions: int
    ):
        self.epsilon = initial_epsilon
        self.epsilon_divider = epsilon_divider
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.neurons_first_layer = neurons_first_layer
        self.neurons_second_layer = neurons_second_layer
        self.input_dimention = input_dimention
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]
        
        self.model = self.__build_model()
        self.replay_buffer = ReplayBuffer(self.memory_size)

    def __build_model(self):
        # NN architecture: (s) -> Q(s, a)^|A| 
        model = Sequential()
        model.add(Dense(self.neurons_first_layer, input_shape=(self.input_dimention,), activation='relu'))
        model.add(Dense(self.neurons_second_layer, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(optimizer='sgd', loss='mse')

        return model

    def load_model(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save_weights(path)

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.store_transition(state, action, reward, next_state)

    def choose_action(self, state, action_space):
        if np.random.random() < self.epsilon:
            action = np.random.choice(action_space)    # random action
        else:
            action = int(np.argmax(self.model.predict(np.array([state]), verbose=0)))    # best action
        return action

    def learn(self):
        if self.replay_buffer.mem_counter > self.batch_size:
            state, action, reward, new_state = self.replay_buffer.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space)
            action_indices = np.dot(action, action_values)

            rewards = self.model.predict(state)

            next_rewards = self.model.predict(new_state)

            q_target = rewards.copy()

            batch_index = np.arange(self.batch_size)

            q_target[batch_index, action_indices] = reward + self.discount_factor * np.max(next_rewards)

            _ = self.model.fit(state, q_target, verbose=0)

            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon/self.epsilon_divider)