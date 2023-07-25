from replay_buffer import ReplayBuffer

def test_get_transictions():
    replay_buffer_size = 50 # Max batch size of past experience
    replay_buffer = ReplayBuffer(replay_buffer_size)
    for _ in range(20):
        replay_buffer.store_transition(1, 1, 1, 1)

    states, actions, rewards, new_states = replay_buffer.get_transitions()

    # check if the number of entries is correct
    assert len(states) == 20
    assert len(actions) == 20
    assert len(rewards) == 20
    assert len(new_states) == 20

    # check if the entries are correct
    assert all(states == 1)
    assert all(actions == 1)
    assert all(rewards == 1)
    assert all(new_states == 1)
    
def test_sample_buffer():
    replay_buffer_size = 50 # Max batch size of past experience
    replay_buffer = ReplayBuffer(replay_buffer_size)
    for _ in range(20):
        replay_buffer.store_transition(1, 1, 1, 1)

    states, actions, rewards, new_states = replay_buffer.sample_buffer(20)

    # check if the number of entries is correct
    assert len(states) == 20
    assert len(actions) == 20
    assert len(rewards) == 20
    assert len(new_states) == 20

    # check if the entries are correct
    assert all(states == 1)
    assert all(actions == 1)
    assert all(rewards == 1)
    assert all(new_states == 1)

if __name__ == "__main__":
    test_get_transictions()
    test_sample_buffer()
    print("All tests passed!")