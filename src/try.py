import numpy as np
from replay_buffer import ReplayBuffer
from keras.models import Sequential
from keras.layers import Dense


buffer = ReplayBuffer(100)
batch_size = 10
action_space = [i for i in range(6)]

model = Sequential()
model.add(Dense(5, input_shape=(1,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(6, activation='linear'))
model.compile(optimizer='sgd', loss='mse')

# insert experience, 20 times
for i in range(11):
    buffer.store_transition(i, i%6, i, i)


# check if there is enough experience
if buffer.mem_counter > batch_size:
    # sample batch
    state, action, reward, new_state = buffer.sample_buffer(batch_size)

    rewards = model.predict(state) # get actual rewards

    next_rewards = model.predict(new_state) # get rewards for next state batch

    q_target = rewards.copy() # copy rewards in order to update only the Q value of the selected action

    batch_index = np.arange(batch_size) # get the indices of the batch

    q_target[batch_index, action] = reward + 0.95 * np.max(next_rewards, axis=1) # Update only the Q-value of selected action

    _ = model.fit(state, q_target, verbose=0)

else:
    print("not enough experience")