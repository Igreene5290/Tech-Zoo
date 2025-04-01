import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
EPISODES = 1000
MAX_STEPS = 500
HIDDEN_SIZE = 128
RENDER_EVERY = 50

# Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU()
        )
        self.actor = nn.Linear(HIDDEN_SIZE, action_dim)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        
    def forward(self, x):
        x = self.shared(x)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

# Environment
env = gym.make("CartPole-v1", render_mode="human" if RENDER_EVERY > 0 else None)
model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam([
    {'params': model.actor.parameters(), 'lr': LR_ACTOR},
    {'params': model.critic.parameters(), 'lr': LR_CRITIC}
])

# Training
for episode in range(EPISODES):
    state, _ = env.reset()
    episode_rewards = []
    log_probs = []
    values = []
    rewards = []
    
    for t in range(MAX_STEPS):
        if episode % RENDER_EVERY == 0:
            env.render()
        
        # Get action and value
        action_probs, value = model(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        next_state, reward, done, _, _ = env.step(action.item())
        
        # Store data
        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)
        state = next_state
        
        if done:
            break
    
    # Calculate returns and advantages
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    
    returns = torch.FloatTensor(returns)
    values = torch.cat(values)
    advantages = returns - values.squeeze()
    
    # Losses
    actor_loss = (-torch.stack(log_probs) * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    total_loss = actor_loss + critic_loss
    
    # Update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {sum(rewards)}")

env.close()