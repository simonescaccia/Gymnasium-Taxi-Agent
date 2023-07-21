import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import gymnasium as gym

np.set_printoptions(threshold=sys.maxsize)

# SETTINGS
TRAIN_MODEL = True         # train the model from q-table or load weights
SARSA_TRAINING = False      # use SARSA Q-learning
N_EPISODES = 10000
N_TEST_EPISODES = 5
TEST_MODEL = True
FILE_POSTFIX = "_" + ("sarsa_" if SARSA_TRAINING else "") + str(N_EPISODES)

# hyperparameters
n_epochs = 1000
batch_size = 100
learning_rate = 0.01
scale_range_x = (-0.5, 0.5)
scale_range_y = (-0.5, 0.5)

def get_train_and_scalers():
    # load q-table experience as numpy array from npy file
    q_table = np.load("q-tables/q_table{}.npy".format(FILE_POSTFIX))

    # transform the q-table into a trainable dataset for a neural network
    # the input is state and action
    # the q-value is the target value for the neural network
    x_train = np.empty((0, 2), int)
    y_train = np.array([])
    for state in range(500):
        if sum(q_table[state]) == 0:
            continue
        for action in range(6):
            x_train = np.append(x_train, [[state, action]], axis=0)
            y_train = np.append(y_train, [q_table[state, action]], axis=0) 

    x_train = x_train.reshape(len(x_train), 2)
    y_train = y_train.reshape(len(y_train), 1)

    # get the scaler of the data to be between 0 and 1
    scale_x = MinMaxScaler(feature_range=scale_range_x)
    scale_y = MinMaxScaler(feature_range=scale_range_y)
    x_train = scale_x.fit_transform(x_train)
    y_train = scale_y.fit_transform(y_train)

    return x_train, y_train, scale_x, scale_y

def train_model(model: Sequential):
    
    x_train, y_train, _, _ = get_train_and_scalers()

    # train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=2)
    model.save_weights("nn_weights/weights{}_{}/cp.ckpt".format(FILE_POSTFIX, n_epochs))

    return model

def load_weights(model):
    model.load_weights("nn_weights/weights{}_{}/cp.ckpt".format(FILE_POSTFIX, n_epochs))
    return model

def test_model(model: Sequential, env):

    _, _, scale_x, _ = get_train_and_scalers()

    # test the model
    for _ in range(N_TEST_EPISODES):
        obs, _ = env.reset()
        done = False

        # play one episode
        while not done:
            # get the best action from the model
            x_test = np.empty((0, 2), int)
            for action in range(6):
                x_test = np.append(x_test, [[obs, action]], axis=0)
            x_test = x_test.reshape(len(x_test), 2)
            # scale the data
            x_test = scale_x.transform(x_test)

            # predict the q-values    
            y_pred = model.predict(x_test, verbose=0)

            action = np.argmax(y_pred)
            print("action: ", action)

            # perform the action on the environment
            next_obs, _, terminated, truncated, _ = env.step(action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

if "__main__" == __name__:
    # NN architecture using TensorFlow
    model = Sequential()
    model.add(Dense(10, input_shape=(2,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    print(model.summary())

    if TRAIN_MODEL:
        model = train_model(model)
    else:
        model = load_weights(model)

    if TEST_MODEL:
        env = gym.make('Taxi-v3', render_mode="human")
        test_model(model, env)




