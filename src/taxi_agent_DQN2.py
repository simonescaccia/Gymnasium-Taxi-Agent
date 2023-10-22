from collections import deque
import random

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class TaxiAgentDQN:
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_decrement: float,
        final_epsilon: float,
        discount_factor: float,
        memory_size: int,
        episode_memory_size: int,
        batch_size: int,
        neurons_first_layer: int,
        neurons_second_layer: int,
        input_dimention: int,
        num_actions: int,
        learning_rate: float
    ):
        self.epsilon = initial_epsilon
        self.epsilon_decrement = epsilon_decrement
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.episode_memory_size = episode_memory_size
        self.batch_size = batch_size
        self.neurons_first_layer = neurons_first_layer
        self.neurons_second_layer = neurons_second_layer
        self.input_dimention = input_dimention
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]
        self.learning_rate = learning_rate
        
        self.model = self.__build_model()
        self.replay_buffer = deque([], maxlen=self.memory_size)
        self.temporary_replay_buffer = deque([], maxlen=self.batch_size)

    def __build_model(self):

        # DQN architecture
        model = Sequential()
        model.add(Dense(self.neurons_first_layer, input_dim=self.input_dimention, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.neurons_second_layer, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.neurons_second_layer, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.num_actions, activation='linear', kernel_initializer='he_uniform'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return model
    
    def init_temporary_remember(self):
        self.temporary_replay_buffer = deque([], maxlen=self.batch_size)
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)    # random action
        else:       
            state = np.reshape(state, [1, self.input_dimention])
            action_values = self.model.predict_on_batch(state)
            action = int(np.argmax(action_values))    # best action
        return action
    
    def temporary_store_transition(self, state, action, reward, new_state, terminated):
        self.temporary_replay_buffer.append((state, action, reward, new_state, terminated))

    def store_transition(self, state, action, reward, new_state, terminated):
        self.replay_buffer.append((state, action, reward, new_state, terminated))

    def store_temporary_transitions(self):
        # store temporary transitions in replay buffer
        while len(self.temporary_replay_buffer) > 0:
            state, action, reward, new_state, terminated = self.temporary_replay_buffer.popleft()
            self.replay_buffer.append((state, action, reward, new_state, terminated))

    def learn(self):
        # check if there is enough experience
        if len(self.replay_buffer) > self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size) # sample a batch of experience
            batch = np.array(batch, dtype=object) # prepare dataset for training

            states = np.array([x[0] for x in batch]).reshape(self.batch_size, self.input_dimention)
            new_states = np.array([x[3] for x in batch]).reshape(self.batch_size, self.input_dimention)

            q_eval = self.model.predict_on_batch(states) # Q(s,a) for all actions
            q_next = self.model.predict_on_batch(new_states) # Q(s',a) for all actions

            q_target = q_eval.copy() # target Q(s,a) = Q(s,a) for all actions

            for i, (_, action, reward, _, terminated) in enumerate(batch):
                q_target[i][action] = reward + self.discount_factor * np.max(q_next[i]) * (1 - terminated) # Q(s,a) <- r + gamma * max_a' Q(s',a') only for non-terminal states
            
            self.model.fit(states, q_target, verbose=0)
        
            self.decrease_epsilon(1)

            
    def decrease_epsilon(self, episodes: int):
        self.epsilon = max(self.final_epsilon, self.epsilon * (self.epsilon_decrement ** episodes)) # decrease epsilon

    def load_model(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save_weights(path)






