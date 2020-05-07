import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
# Discounting of future actions versus current reward
DISCOUNT = 0.95
EPISODES = 2_500
SHOW_EVERY = 2000 # Show the status every this many episodes
# Discretize the number of possible states so it is not continuous
# Note that most RL problems the state space is not known beforehand like this
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
EPSILON = 0.5 # Controls the randomness/ exploratory-ness of the agent
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# The q-table has tuples with each state and the q-value for being in each of those states
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Convert the current state into a discretized version
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    return tuple(discrete_state.astype(np.int))

# There are two observed states: velocity, and momentum
print(env.observation_space.high)
print(env.observation_space.low)
# The three actions are: accelerate left, stop, accelerate right
print(env.action_space.n)

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
        
    done = False
    discrete_state = get_discrete_state(env.reset()) # env.reset() returns the original state
    while not done:
        if np.random.random() > EPSILON:
            # Find the action to take by picking the highest q value, using the current state as an index
            action = np.argmax(q_table[discrete_state]) 
        else:
            # Take a random action
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0 # Reward for completing the gym
            
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= EPSILON_DECAY_VALUE
    
env.close()