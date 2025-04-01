import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym

# Define Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)  # Action probabilities
        self.critic = nn.Linear(128, 1)  # State value

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

# Worker Process for A3C Training
def train_worker(worker_id, global_model, optimizer):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    local_model = ActorCritic(state_dim, action_dim)
    
    while True:
        state = env.reset()
        log_probs = []
        values = []
        rewards = []

        for _ in range(100):  # Collect experience
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, value = local_model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, done, _ = env.step(action)
            log_probs.append(torch.log(action_probs[action]))
            values.append(value)
            rewards.append(reward)
            
            if done:
                break
            state = next_state
        
        # Compute Advantage
        R = 0 if done else local_model(torch.tensor(next_state, dtype=torch.float32))[1].item()
        returns = []
        for reward in reversed(rewards):
            R = reward + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        # Compute Loss
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    state_dim = 4  # CartPole state size
    action_dim = 2  # Left or Right

    global_model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    processes = []
    for worker_id in range(mp.cpu_count()):  # Use all CPU cores
        p = mp.Process(target=train_worker, args=(worker_id, global_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
