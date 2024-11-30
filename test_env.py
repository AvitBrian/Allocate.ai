from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy  # Use a greedy policy for testing
from rl.memory import SequentialMemory
from resource_allocation_env import ResourceAllocationEnv
import tensorflow as tf

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = GreedyQPolicy()  # Use GreedyQPolicy for deterministic action selection
    memory = SequentialMemory(limit=2000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    # Initialize the environment
    env = ResourceAllocationEnv(grid_size=5, resources=50)
    states = env.observation_space.shape
    actions = env.action_space.n

    # Build and summarize the model
    model = build_model(states, actions)
    model.summary()

    # Build the DQN agent
    dqn = build_agent(model, actions)

    # Load the pre-trained weights (if available)
    try:
        dqn.load_weights('Policies/resource_allocation_dqn_weights.h5')
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")

    # Compile the agent
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Test the agent
    dqn.test(env, nb_episodes=1, visualize=True)
