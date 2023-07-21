# array of 6 rewards
import numpy as np


my_rewards = np.array([1, 2, 3, 4, 5, 6])
print(my_rewards)

all_rewards = np.array([[]])
# new array
new_rewards = np.array([])
# append to new array 10
new_rewards = np.append(new_rewards, [10])
print(new_rewards)

# for each reward in my_rewards append to new_rewards
for reward in my_rewards:
    new_rewards = np.append(new_rewards, [reward])

print(new_rewards)
# append new_rewards to all_rewards
all_rewards = np.append(all_rewards, [new_rewards]).reshape(-1, 7)
print(all_rewards)

print("obs: ", all_rewards[:,0])
print("reward: ", all_rewards[:,1:7])


epsilon = 1

def decrease():
    # decrease epsilon: prefer exploration first, then exploitation
    global epsilon
    epsilon = max(0.1, epsilon - epsilon/100)

print(epsilon)
decrease()
print(epsilon)
