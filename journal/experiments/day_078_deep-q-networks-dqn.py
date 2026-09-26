import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 10
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
EPISODES = 500
MAX_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(DEVICE),
            torch.LongTensor(action).to(DEVICE),
            torch.FloatTensor(reward).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.action_dim = action_dim
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if not eval_mode and random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.policy_net(state).argmax().item()

    def optimize(self):
        if len(self.buffer) < MIN_REPLAY_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []
    losses = []

    print(f"Training on {DEVICE} | Env: {ENV_NAME}")
    for ep in range(EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        ep_losses = []
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            loss = agent.optimize()
            if loss:
                ep_losses.append(loss)
            if done:
                break
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
        episode_rewards.append(ep_reward)
        losses.append(np.mean(ep_losses) if ep_losses else 0)
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Ep {ep+1:4d} | Avg Reward (50): {avg_reward:.2f} | Eps: {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * agent.steps_done / EPS_DECAY):.3f}")
    env.close()
    return agent, episode_rewards, losses

def evaluate(agent, n_episodes=10):
    env = gym.make(ENV_NAME, render_mode="human")
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    env.close()
    print(f"Evaluation over {n_episodes} episodes: Mean Reward = {np.mean(rewards):.2f}")

def plot_results(rewards, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    ax1.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), color='red', label='Moving Avg (50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(losses, alpha=0.6, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("dqn_cartpole_results.png")
    print("Plot saved to dqn_cartpole_results.png")

if __name__ == "__main__":
    agent, rewards, losses = train()
    plot_results(rewards, losses)
    evaluate(agent, n_episodes=5)
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
    print("Model saved to dqn_cartpole.pth")