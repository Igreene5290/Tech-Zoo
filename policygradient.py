import gymnasium as gym
import numpy as np
import time  # Added for frame rate control

# Hyperparameters
LEARNING_RATE = 0.01
GAMMA = 0.99
EPISODES = 1000
RENDER_EVERY = 50  # Render every N episodes
FRAME_DELAY = 0.02  # Seconds between frames (controls speed)

class Policy:
    def __init__(self, state_dim, action_dim):
        self.weights = np.random.rand(state_dim, action_dim) * 0.01
    
    def forward(self, state):
        z = np.dot(state, self.weights)
        return np.exp(z) / np.sum(np.exp(z))
    
    def act(self, state):
        probs = self.forward(state)
        return np.random.choice(len(probs), p=probs), probs

# Initialize with rendering
env = gym.make("CartPole-v1", render_mode="human")  # <-- Enable human rendering
policy = Policy(env.observation_space.shape[0], env.action_space.n)

for episode in range(EPISODES):
    state, _ = env.reset()
    episode_data = []
    done = False
    
    while not done:
        # Render logic
        if episode % RENDER_EVERY == 0:
            env.render()
            time.sleep(FRAME_DELAY)  # Control animation speed
        
        action, probs = policy.act(state)
        next_state, reward, done, _, _ = env.step(action)
        
        episode_data.append({
            "state": state,
            "action": action,
            "prob": probs[action],
            "reward": reward
        })
        state = next_state
    
    # Policy update (same as before)
    # ... [rest of training logic] ...

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {sum(d['reward'] for d in episode_data)}")

env.close()  # Close render window