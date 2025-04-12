import yaml
import os
import matplotlib.pyplot as plt
import torch

CONFIG_FILE = "dqn.yml"

def setup_logging(log_dir):
    """
    Set up logging to a file.
    :param log_path: Path to the log file.
    """
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Create the run log directory: if log_dir is empty append _1, else append _last + 1
    prefix = "taxi-v3"
    if not os.listdir(log_dir):
        run_dir = os.path.join(log_dir, f"{prefix}_1")
    else:
        dirs = os.listdir(log_dir)
        dirs = sorted(dirs)
        last_dir = dirs[-1]
        last_num = int(last_dir.split('_')[-1])
        run_dir = os.path.join(log_dir, f"{prefix}_{last_num + 1}")
    os.makedirs(run_dir)
    return run_dir


def load_config(log_dir, config_path):
    """
    Load the configuration file.
    :param config_path: Path to the YAML configuration file.
    :return: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Save the configuration to the log directory
    config_file_path = os.path.join(log_dir, CONFIG_FILE)
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file)

    return config

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = torch.optim.Adam

class DQN():
    def __init__(self, config, log_dir):
        self.exploration_fraction = config['exploration_fraction']
        self.exploration_final_eps = config['exploration_final_eps']
        self.n_timestep = int(config['n_timesteps'])


        # Logs
        self.log_dir = log_dir
        self.history_epsilon = []
        
    def get_epsilon(self, current_timestep):
        """
        Get the current epsilon value.
        Epsilon is linearly decayed from 1.0 to exploration_final_eps in the first exploration_fraction timesteps.
        After that, it is kept constant.
        :return: Current epsilon value.
        """
        current_exploration_fraction = current_timestep / self.n_timestep
        if current_exploration_fraction <= self.exploration_fraction:
            # y = mx + q
            # p1 = (0, 1.0), p2 = (exploration_fraction*n_timesteps, exploration_final_eps)
            # m = (y1 - y2) / (x1 - x2) = (exploration_final_eps - 1.0) / exploration_fraction*n_timesteps
            m = (self.exploration_final_eps - 1.0) / (self.exploration_fraction * self.n_timestep)
            # q = (x1*y2 - x2*y1) / (x1 - x2) = -exploration_fraction*n_timesteps / -exploration_fraction*n_timesteps = 1
            q = 1.0
            return m * current_timestep + q
        else:
            return self.exploration_final_eps
        
    def train(self):
        for timestep in range(self.n_timestep):
            epsilon = self.get_epsilon(timestep)
            self.history_epsilon.append(epsilon)

    def save_logs(self):
        # Plot epsilon
        plt.plot(self.history_epsilon)
        plt.title('Epsilon')
        plt.xlabel('Timesteps')
        plt.ylabel('Epsilon')
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, "epsilon.png"))
        plt.close()

if __name__ == "__main__":
    # Set up log files
    log_dir = setup_logging("logs")

    # Load dqn.yml
    config = load_config(log_dir, CONFIG_FILE)

    # Create DQN agent
    agent = DQN(config, log_dir)

    # Train the agent
    agent.train()

    # Save logs
    agent.save_logs()