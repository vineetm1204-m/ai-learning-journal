import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import gymnasium as gym
import matplotlib.pyplot as plt

# ==========================================
# Configuration & Hyperparameters
# ==========================================
ENV_ID = "CartPole-v1"
SEED = 42
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
N_EPISODES = 500
MAX_T = 1000
SOLVED_SCORE = 475.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# Reproducibility
# ==========================================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# ==========================================
# Neural Network: Q-Network (MLP)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# Replay Buffer
# ==========================================
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# ==========================================
# DQN Agent
# ==========================================
class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
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

    def act(self, state, eps=None):
        if eps is None:
            eps = self.eps
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
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

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

# ==========================================
# Training Loop
# ==========================================
def train_dqn():
    env = gym.make(ENV_ID)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size, SEED)

    scores = []
    scores_window = deque(maxlen=100)
    eps_history = []

    print(f"\nStarting Training on {ENV_ID}...")
    print(f"State Size: {state_size}, Action Size: {action_size}")

    for i_episode in range(1, N_EPISODES + 1):
        state, _ = env.reset(seed=SEED + i_episode)
        score = 0
        for t in range(MAX_T):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps_history.append(agent.eps)
        agent.decay_epsilon()

        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.eps:.4f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= SOLVED_SCORE:
            print(f"\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_dqn.pth')
            break

    env.close()
    return scores, eps_history

# ==========================================
# Evaluation & Visualization
# ==========================================
def evaluate_agent(agent, n_episodes=5, render=False):
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    eval_scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        eval_scores.append(total_reward)
    env.close()
    return np.mean(eval_scores)

def plot_results(scores, eps_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(np.arange(len(scores)), scores, label='Episode Score', alpha=0.6)
    ax1.plot(np.arange(len(scores)), np.convolve(scores, np.ones(100)/100, mode='valid'), label='Moving Avg (100)', color='red')
    ax1.axhline(y=SOLVED_SCORE, color='g', linestyle='--', label=f'Solved Threshold ({SOLVED_SCORE})')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Episode')
    ax1.set_title('DQN Training Performance (CartPole-v1)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(np.arange(len(eps_history)), eps_history, color='orange')
    ax2.set_ylabel('Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_title('Epsilon Decay Schedule')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('dqn_training_results.png')
    print("\nPlot saved to 'dqn_training_results.png'")
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    scores, eps_history = train_dqn()

    # Load best model for evaluation
    env = gym.make(ENV_ID)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, SEED)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint_dqn.pth', map_location=DEVICE))

    avg_score = evaluate_agent(agent, n_episodes=10, render=False)
    print(f"\nFinal Evaluation (10 episodes, greedy policy): Average Score = {avg_score:.2f}")

    plot_results(scores, eps_history)