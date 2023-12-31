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
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        memory_size: int,
        batch_size: int,
        neurons_first_layer: int,
        input_dimention: int,
        num_actions: int,
        learning_rate: float
    ):
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.neurons_first_layer = neurons_first_layer
        self.input_dimention = input_dimention
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]
        self.learning_rate = learning_rate
        
        self.model = self.__build_model()
        self.replay_buffer = ReplayBuffer(self.memory_size)

    def __build_model(self):
        # NN architecture: (s) -> Q(s, a)^|A| 
        model = Sequential()
        model.add(Dense(self.input_dimention, input_shape=(self.input_dimention,)))
        model.add(Dense(self.neurons_first_layer, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def load_model(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save_weights(path)

    def remember(self, state, action, reward, next_state, terminated):
        self.replay_buffer.store_transition(state, action, reward, next_state, terminated)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)    # random action
            #print("random action" + str(action))
        else:       
            prediction = self.model.predict_on_batch(np.array([state]))
            action = int(np.argmax(prediction[0]))    # best action
        return action

    def learn(self):
        # check if there is enough experience
        if self.replay_buffer.mem_counter > self.batch_size:
            state, action, reward, new_state, terminated = self.replay_buffer.sample_buffer(self.batch_size) # sample batch

            rewards = self.model.predict_on_batch(state) # get actual rewards

            next_rewards = self.model.predict_on_batch(new_state) # get rewards for next state batch

            q_target = rewards.copy() # copy rewards in order to update only the Q value of the selected action

            batch_index = np.arange(self.batch_size) # get the indices of the batch

            q_target[batch_index, action] = reward + self.discount_factor * np.max(next_rewards, axis=1) * (1 - terminated) # Update only the Q-value of selected action

            _ = self.model.fit(state, q_target, verbose=0)

    def decay_epsilon(self):
        # decrease epsilon: prefer exploration first, then exploitation 
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)