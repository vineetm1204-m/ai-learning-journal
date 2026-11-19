import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME = "CartPole-v1"
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TAU = 1e-3
UPDATE_EVERY = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MAX_EPISODES = 500
MAX_STEPS = 500
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=42):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.eps = EPS_START

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

# Training Loop
def train():
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, SEED)

    scores = []
    scores_window = deque(maxlen=100)
    eps = EPS_START

    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=SEED)
        score = 0
        for t in range(MAX_STEPS):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(EPS_END, eps * EPS_DECAY)
        agent.decay_epsilon()

        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= 475.0:
            print(f"\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(agent.qnetwork_local.state_dict(), "dqn_checkpoint.pth")
            break
    env.close()
    return scores

# Evaluation
def evaluate(model_path="dqn_checkpoint.pth", n_episodes=10):
    env = gym.make(ENV_NAME, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, SEED)
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
    agent.qnetwork_local.eval()

    for i in range(n_episodes):
        state, _ = env.reset(seed=SEED + i)
        total_reward = 0
        for t in range(MAX_STEPS):
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Evaluation Episode {i+1}: Total Reward = {total_reward}")
    env.close()

# Plotting
def plot_scores(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores, label='Episode Score')
    plt.plot(np.arange(len(scores)), 
             np.convolve(scores, np.ones(100)/100, mode='valid'), 
             label='100-Episode Moving Avg', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Performance on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_training_curve.png")
    plt.show()

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Environment: {ENV_NAME}")
    scores = train()
    plot_scores(scores)
    evaluate()