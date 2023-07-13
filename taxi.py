# Import and create the gymnasium environment
import gymnasium as gym
env = gym.make('Taxi-v3', render_mode="human")

# Reset the environment to get the first observation
# env.reset() start an episode. This function
# resets the environment to a starting position and returns an initial observation. 
# Setting done = False will be useful later to check if a game 
# is terminated (i.e., the player wins or loses).
done = False
observation, info = env.reset()
print("Observation: ", observation)
print("Info: ", info)

# e.g. info {'prob': 1.0, 'action_mask': array([0, 1, 1, 0, 0, 0], dtype=int8)}
# As taxi is not stochastic, the transition probability is always 1.0
# For some cases, taking an action will have no effect on the state of 
# the episode. In v0.25.0, info["action_mask"] contains a np.ndarray for 
# each of the actions specifying if the action will change the state.

# e.g. observation 466
# An observation is returned as an int() that encodes the corresponding state, calculated by 
# ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
# Note that there are 400 states that can actually be reached during an 
# episode. The missing states correspond to situations in which the 
# passenger is at the same location as their destination, as this 
# typically signals the end of an episode. Four additional states can be 
# observed right after a successful episodes, when both the passenger 
# and the taxi are at the destination. This gives a total of 404 reachable 
# discrete states.

# render the enviroment
env.render()

# Execute some random actions in the environment
num_steps = 100
for i in range(100):
    # sample a random action not using a Q-value based algorithm 
    action = env.action_space.sample()
    print("Action: ", action)
    # e.g. action 3 (Move left)

    # perform this action on the environment
    tuple = env.step(action)
    print("Observation: ", tuple[0])
    print("Reward: ", tuple[1])
    print("Action Mask: ", tuple[4]['action_mask'])

    # print the new state
    env.render()

    # Sleep for 500 milliseconds
    import time
    time.sleep(0.5)

env.close()