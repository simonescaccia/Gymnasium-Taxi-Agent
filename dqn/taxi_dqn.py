from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
# Define model
class Net(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
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
        self.s_0, _ = self.env.reset() # Initial state
        self.num_states = 500
        self.episodes = 10000
        self.batch_size = 64
        self.loss_function = nn.MSELoss()

        replay_memory_size = 5000
        replay_memory_burn_in = 1000
        learning_rate = 0.001
        self.epsilon = 1 # 1 = 100% random actions
        in_states = self.num_states
        h1_nodes = 128
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
            self.s_0, _ = self.env.reset()
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
            self.s_0, _ = self.env.reset()

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
                    self.epsilon = max(self.epsilon - 1/self.episodes, 0)
                    self.epsilon_history.append(self.epsilon)
                    
                    # Log training progress
                    self.training_rewards.append(self.rewards)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []

                    print(
                        "\nEpisode {:d}  Episode reward = {:.2f}   Mean loss = {:.2f}".format(
                            ep, self.rewards, np.mean(self.update_loss)), end="")

                    if ep >= self.episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
        # save models
        self.save_models()
        # plot
        self.plot_training_rewards()

    def save_models(self):
        torch.save(self.policy_dqn, "Q_net")

    def load_models(self):
        self.policy_dqn = torch.load("Q_net")
        self.policy_dqn.eval()

    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()


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


    def evaluate(self, eval_env):
        done = False
        s, _ = eval_env.reset()
        rew = 0
        while not done:
            action = self.policy_dqn.greedy_action(torch.FloatTensor(s).to(device))
            s, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            rew += r

        print("Evaluation cumulative reward: ", rew)


if __name__ == '__main__':
    agent = TaxiDQN()

    agent.train()

    agent.env = gym.make('Taxi-v3', render='human')
    agent.evaluate()