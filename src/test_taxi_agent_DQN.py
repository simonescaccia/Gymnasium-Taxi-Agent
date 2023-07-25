import gymnasium as gym
from taxi_agent_DQN import TaxiAgentDQN

# hyperparameters
replay_buffer_size = 10000 # Max batch size of past experience
temporary_replay_buffer_size = 200 # Max number of steps in an episode
batch_size = 64 # Training set size
start_epsilon = 1
epsilon_decrement = 0.999
final_epsilon = 0.01
discount_factor = 0.99
neurons_first_layer = 64
neurons_second_layer = 64
learning_rate = 0.005



def test_remember_last_episode():
    taxi_agent = TaxiAgentDQN(
        initial_epsilon=start_epsilon,
        epsilon_decrement=epsilon_decrement,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        memory_size=replay_buffer_size,
        temporary_memory_size=temporary_replay_buffer_size,
        batch_size=batch_size,
        neurons_first_layer=neurons_first_layer,
        neurons_second_layer=neurons_second_layer,
        input_dimention=1,
        num_actions=env.action_space.n,
        learning_rate=learning_rate
    )

    repeat = 5
    for _ in range(repeat):
        taxi_agent.init_temporary_remember()
        for _ in range(20):
            taxi_agent.temporary_remember(1, 1, 1, 1)    
        taxi_agent.remember_last_episode()


    states, actions, rewards, new_states = taxi_agent.replay_buffer.get_transitions()

    # check if the number of entries is correct
    assert len(states) == 20 * repeat
    assert len(actions) == 20 * repeat
    assert len(rewards) == 20 * repeat
    assert len(new_states) == 20 * repeat

    # check if the entries are correct
    assert all(states == 1)
    assert all(actions == 1)
    assert all(rewards == 1)
    assert all(new_states == 1)


if __name__ == "__main__":

    env = gym.make('Taxi-v3')

    test_remember_last_episode()
    print("All tests passed!")

