from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
# Define model
class Net(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)     # second fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output
        return x
    
# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen, burn_in):
        self.memory = deque(maxlen=maxlen)
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward', 'done', 'next_state'])

    
    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.memory[i] for i in samples])
        return batch

    def append(self, s_0, a, r, d, s_1):
        self.memory.append(
            self.Buffer(s_0, a, r, d, s_1))
    
    # Burn-in capacity: < 1 means we need more data to start training, >= 1 means we have enough data
    def burn_in_capacity(self):
        return len(self.memory) / self.burn_in

    def __len__(self):
        return len(self.memory)
    

class TaxiDQN:

    def __init__(self):

        self.env = gym.make('Taxi-v3')

        self.step_count = 0
        self.gamma = 0.99
        self.s_0, _ = self.env.reset(seed = 0) # Initial state
        self.num_states = 500
        self.episodes = 13000
        self.batch_size = 256
        self.loss_function = nn.MSELoss()

        replay_memory_size = 100000
        replay_memory_burn_in = 100000
        learning_rate = 0.003
        self.epsilon = 1 # 1 = 100% random actions
        in_states = self.num_states
        h1_nodes = 256
        out_actions = 6

        self.policy_dqn = Net(in_states, h1_nodes, out_actions).to(device)
        self.target_dqn = Net(in_states, h1_nodes, out_actions).to(device)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        self.memory = ReplayMemory(replay_memory_size, replay_memory_burn_in)

        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=learning_rate)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        self.rewards_per_episode = np.zeros(self.episodes)

        # List to keep track of epsilon decay
        self.epsilon_history = []

        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.rewards = 0


    def states_to_onehot(self, states: torch.IntTensor):
        # Convert state to one-hot encoding
        onehot = torch.zeros((states.shape[0], self.num_states)).to(device)
        onehot[torch.arange(states.shape[0]), states.squeeze()] = 1
        return onehot

    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.IntTensor([self.s_0]).to(device)
                action = self.policy_dqn(self.states_to_onehot(state)).argmax().item()

        #simulate action
        s_1, r, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        #put experience in the buffer
        self.memory.append(self.s_0, action, r, terminated, s_1)

        self.rewards += r

        self.s_0 = s_1

        self.step_count += 1
        if done:
            self.s_0, _ = self.env.reset(seed=0)
        return done

    # Implement DQN training algorithm
    def train(self):

        # Populate replay buffer
        print("Populating replay buffer...")
        while self.memory.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        print("Replay buffer populated.")

        ep = 0
        training = True
        while training:
            # if ep > 0 and (ep % 1000) == 0:
            #     self.env = gym.make('Taxi-v3', render_mode="human")
            # elif ep > 1 and (ep % 1000) == 1:
            #     self.env = gym.make('Taxi-v3')
            self.s_0, _ = self.env.reset(seed = 0)

            self.rewards = 0
            done = False
            while not done:
                # if ((ep % 5) == 0):
                #     self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                else:
                    done = self.take_step(mode='exploit')

                self.update_network()

                if done:
                    ep += 1
                    
                    # Decay epsilon
                    if self.epsilon >= 0.02:
                        self.epsilon = self.epsilon - self.epsilon * 0.01
                    # self.epsilon = max(self.epsilon - 1/self.episodes, 0)
                    self.epsilon_history.append(self.epsilon)
                    
                    # Log training progress
                    ep_loss = np.mean(self.update_loss)
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(ep_loss)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(ep_loss)

                    print(
                        "\nEpisode {:d}  Episode reward = {:.2f}  Epsilon = {:.2f}  Episode length = {:.0f}  Mean last 100 rewards = {:.2f}".format(
                            ep, self.rewards, self.epsilon, len(self.update_loss), np.mean(self.training_rewards[-100:])), end="")

                    self.update_loss = []
                    if ep >= self.episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
        # save models
        self.save_models()
        # plot
        self.plot_training_info()

    def save_models(self):
        torch.save(self.policy_dqn, "Q_net")

    def load_models(self):
        self.policy_dqn = torch.load("Q_net", weights_only=False)
        self.policy_dqn.eval()

    def plot_training_info(self, save=False):
        # Plot three figures in the same plot: training rewards, loss and epsilon decay
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(self.training_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(self.training_loss)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(self.epsilon_history)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid()

        if save:
            plt.savefig('training_info.png')
        else:
            plt.show()      

    def calculate_loss(self, batch):
        #extract info from batch
        states, actions, rewards, dones, next_states = batch

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(device)
        states = torch.IntTensor(states).reshape(-1, 1).to(device)
        next_states = torch.IntTensor(next_states).reshape(-1, 1).to(device)

        ###############
        # DQN Update #
        ###############
        qvals = self.policy_dqn(self.states_to_onehot(states))
        qvals = torch.gather(qvals, 1, actions)

        next_qvals = self.target_dqn(self.states_to_onehot(next_states))
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        # loss = self.loss_function( Q(s,a) , target_Q(s,a))
        loss = self.loss_function(qvals, target_qvals)

        return loss


    def update_network(self):
        batch = self.memory.sample_batch(self.batch_size)
        loss = self.calculate_loss(batch)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_loss.append(loss.item())


    def evaluate(self):
        done = False
        s, _ = self.env.reset(seed = 0)
        rew = 0
        while not done:
            with torch.no_grad():
                state = torch.IntTensor([s]).to(device)
                action = self.policy_dqn(self.states_to_onehot(state)).argmax().item()
            s, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            rew += r

        print("Evaluation cumulative reward: ", rew)


if __name__ == '__main__':
    print("Device: ", device)

    agent = TaxiDQN()
    train_test = int(sys.argv[1])

    if train_test == 0:
        agent.train()
    elif train_test == 1:
        agent.load_models()
    else:
        raise ValueError("Train: 0, Train: 1")

    agent.env = gym.make('Taxi-v3', render_mode="human")
    agent.evaluate()