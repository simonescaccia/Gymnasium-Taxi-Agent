# Gymnasium Taxi Agent

## 1 Problem Description

The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
off at one of four locations.

## 1.1 Description of the environment

There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
5x5 grid world. The taxi starts at a random square and the passenger at one of the
designated locations. The goal is to move the taxi to the passenger's location, pick up the passenger,
move to the passenger's desired destination, and
drop off the passenger. Once the passenger is dropped off, the episode ends.

The player receives positive rewards for successfully dropping off the passenger at the correct
location. Negative rewards for incorrect attempts to pick up/drop off the passenger and
for each step where another reward is not received.

Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

## 1.2 Action Space

The action shape is `(1,)` in the range `{0, 5}` indicating
which direction to move the taxi or to pick up/drop off the passenger.

- 0: Move south (down)
- 1: Move north (up)
- 2: Move east (right)
- 3: Move west (left)
- 4: Pickup passenger
- 5: Drop off the passenger

## 1.3 Observation Space

There are 500 discrete states since there are 25 taxi positions, 5 possible
locations of the passenger (including the case when the passenger is in the
taxi), and 4 destination locations.

Destinations on the map are represented with the first letter of the color.

Passenger locations:

- 0: Red
- 1: Green
- 2: Yellow
- 3: Blue
- 4: In the taxi

Destinations:

- 0: Red
- 1: Green
- 2: Yellow
- 3: Blue

An observation is returned as an `int()` that encodes the corresponding state, calculated by
`((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`

Note that there are only 400 states that can be reached during an
episode. The missing states correspond to situations in which the passenger
is at the same location as their destination, as this typically signals the
end of an episode. Four additional states can be observed right after a
successful episode when both the passenger and the taxi are at the destination.
This gives a total of 404 reachable discrete states.

## 1.4. Starting State

The episode starts with the player in a random state.

## 1.5 Rewards

- -1 per step unless another reward is triggered.
- +20 delivering the passenger.
- -10  executing "pickup" and "drop-off" actions illegally.

An action that results in a noop (no operation), like moving into a wall, will incur the time step
penalty. Noops can be avoided by sampling the `action_mask` returned in `info`.

## 1.6 Episode End

The episode ends if the following happens:

- Termination:
        1. The taxi drops off the passenger.
- Truncation (when using the time_limit wrapper):
        1. The length of the episode is 200.

## 1.7 Information

`step()` and `reset()` return a dict with the following keys:

- p - transition probability for the state.
- action_mask - if actions will cause a transition to a new state.

As the taxi is not stochastic, the transition probability is always 1.0.
In some cases, taking action will not affect the state of the episode.
In v0.25.0, ``info["action_mask"]`` contains an ``np.ndarray`` for each of the actions specifying
if the action will change the state.

To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``.
Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.

## 2 Proposed Solution

We will use a Q-learning algorithm that uses a neural network to approximate the Q-values.

### 2.1 Q-learning

#### 2.1.1 Initial considertions

In the proposed solution, we will avoid the facilities provided by the Gymnasium environment to avoid the agent to take bad actions. Indeed, using ``info["action_mask"]`` to choose the next action, it is possible to avoid 'Pickup passenger' and 'Drop off the passenger' actions when they are not possible. As a consequence, our agent can receive the negative reward -10 by executing "pickup" and "drop-off" actions illegally. 

## Resources

### Documentation

[Gymnasium Taxi documentation](https://gymnasium.farama.org/environments/toy_text/taxi/#taxi_ref)

### Dependencies

```python -m pip install gymnasium```
```python -m pip install gymnasium[toy-text]```
