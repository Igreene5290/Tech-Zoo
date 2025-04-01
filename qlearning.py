import gymnasium as gym
import numpy as np
from time import sleep

# Create the environment
env = gym.make("Taxi-v3", render_mode="human")
q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 10_000  # Training episodes

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        next_state, reward, done, _, _ = env.step(action)
        
        # Q-learning update
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
    
    if episode % 1000 == 0:
        print(f"Episode: {episode}")

# Test the trained agent
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()
    sleep(0.5)  # Slow down for visualization

print(f"Total reward: {total_reward}")
env.close()