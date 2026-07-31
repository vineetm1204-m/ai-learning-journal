import numpy as np
import random
from collections import defaultdict

# ============================================================
# Day 36: RL Foundations - MDP, Rewards, Policy
# Self-contained mini-experiment: Grid World with Policy Iteration
# ============================================================

# ------------------------------------------------------------
# 1. MDP Definition: 4x4 Grid World
# ------------------------------------------------------------
class GridWorld:
    def __init__(self, size=4, goal=(3, 3), holes=[(1, 1), (2, 2)], gamma=0.9):
        self.size = size
        self.goal = goal
        self.holes = set(holes)
        self.gamma = gamma
        self.states = [(r, c) for r in range(size) for c in range(size)]
        self.actions = ['up', 'down', 'left', 'right']
        self.action_idx = {a: i for i, a in enumerate(self.actions)}
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Transition dynamics: deterministic with 0.8 success, 0.1 slip left/right
        self.P = self._build_transition_matrix()
        self.R = self._build_reward_matrix()
    
    def _build_transition_matrix(self):
        """P[s][a][s'] = probability"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        slip = {'up': ['left', 'right'], 'down': ['left', 'right'],
                'left': ['up', 'down'], 'right': ['up', 'down']}
        
        for s_idx, (r, c) in enumerate(self.states):
            if (r, c) == self.goal or (r, c) in self.holes:
                # Terminal states: self-loop
                for a in range(self.n_actions):
                    P[s_idx, a, s_idx] = 1.0
                continue
            
            for a_idx, action in enumerate(self.actions):
                # Main direction (0.8)
                dr, dc = moves[action]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    ns_idx = self.state_to_idx[(nr, nc)]
                else:
                    ns_idx = s_idx  # Hit wall, stay
                P[s_idx, a_idx, ns_idx] += 0.8
                
                # Slip directions (0.1 each)
                for slip_action in slip[action]:
                    dr, dc = moves[slip_action]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        ns_idx = self.state_to_idx[(nr, nc)]
                    else:
                        ns_idx = s_idx
                    P[s_idx, a_idx, ns_idx] += 0.1
        return P
    
    def _build_reward_matrix(self):
        """R[s][a] = expected immediate reward"""
        R = np.zeros((self.n_states, self.n_actions))
        for s_idx, (r, c) in enumerate(self.states):
            if (r, c) == self.goal:
                R[s_idx, :] = 10.0
            elif (r, c) in self.holes:
                R[s_idx, :] = -10.0
            else:
                R[s_idx, :] = -0.1  # Small step cost
        return R
    
    def step(self, state, action):
        """Sample next state and reward"""
        s_idx = self.state_to_idx[state]
        a_idx = self.action_idx[action]
        probs = self.P[s_idx, a_idx]
        ns_idx = np.random.choice(self.n_states, p=probs)
        next_state = self.states[ns_idx]
        reward = self.R[s_idx, a_idx]
        done = next_state == self.goal or next_state in self.holes
        return next_state, reward, done
    
    def render_policy(self, policy):
        """Visualize policy"""
        arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        for (r, c), a_idx in policy.items():
            if (r, c) == self.goal:
                grid[r][c] = 'G'
            elif (r, c) in self.holes:
                grid[r][c] = 'H'
            else:
                grid[r][c] = arrows[self.actions[a_idx]]
        print("\nPolicy:")
        for row in grid:
            print(' '.join(f'{c:>2}' for c in row))
        print()

    def render_values(self, V):
        """Visualize value function"""
        grid = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        for (r, c), v in V.items():
            grid[r][c] = v
        print("\nValue Function:")
        for row in grid:
            print(' '.join(f'{v:6.2f}' for v in row))
        print()


# ------------------------------------------------------------
# 2. Policy Evaluation (Iterative)
# ------------------------------------------------------------
def policy_evaluation(env, policy, theta=1e-6, max_iter=1000):
    """Compute V^π for a given policy"""
    V = {s: 0.0 for s in env.states}
    for iteration in range(max_iter):
        delta = 0.0
        for s in env.states:
            if s == env.goal or s in env.holes:
                continue
            s_idx = env.state_to_idx[s]
            a = policy[s]
            a_idx = env.action_idx[a]
            v = 0.0
            for ns_idx, prob in enumerate(env.P[s_idx, a_idx]):
                if prob > 0:
                    ns = env.states[ns_idx]
                    v += prob * (env.R[s_idx, a_idx] + env.gamma * V[ns])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            print(f"Policy Evaluation converged in {iteration+1} iterations (delta={delta:.6f})")
            break
    return V


# ------------------------------------------------------------
# 3. Policy Improvement
# ------------------------------------------------------------
def policy_improvement(env, V, policy):
    """Improve policy greedily w.r.t. V"""
    policy_stable = True
    for s in env.states:
        if s == env.goal or s in env.holes:
            continue
        old_action = policy[s]
        s_idx = env.state_to_idx[s]
        best_value = -np.inf
        best_action = old_action
        for a_idx, action in enumerate(env.actions):
            q = 0.0
            for ns_idx, prob in enumerate(env.P[s_idx, a_idx]):
                if prob > 0:
                    ns = env.states[ns_idx]
                    q += prob * (env.R[s_idx, a_idx] + env.gamma * V[ns])
            if q > best_value:
                best_value = q
                best_action = action
        policy[s] = best_action
        if best_action != old_action:
            policy_stable = False
    return policy, policy_stable


# ------------------------------------------------------------
# 4. Policy Iteration (Full Algorithm)
# ------------------------------------------------------------
def policy_iteration(env):
    """Policy Iteration: Evaluate -> Improve until stable"""
    # Initialize random policy
    policy = {}
    for s in env.states:
        if s == env.goal or s in env.holes:
            policy[s] = 'up'  # Arbitrary for terminal
        else:
            policy[s] = random.choice(env.actions)
    
    print("Initial random policy:")
    env.render_policy(policy)
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Policy Iteration {iteration} ===")
        V = policy_evaluation(env, policy)
        env.render_values(V)
        policy, stable = policy_improvement(env, V, policy)
        env.render_policy(policy)
        if stable:
            print(f"Policy converged after {iteration} iterations!")
            break
    return policy, V


# ------------------------------------------------------------
# 5. Value Iteration (Alternative: combines eval + improve)
# ------------------------------------------------------------
def value_iteration(env, theta=1e-6, max_iter=1000):
    """Value Iteration: directly compute optimal V*"""
    V = {s: 0.0 for s in env.states}
    for iteration in range(max_iter):
        delta = 0.0
        for s in env.states:
            if s == env.goal or s in env.holes:
                continue
            s_idx = env.state_to_idx[s]
            best_value = -np.inf
            for a_idx in range(env.n_actions):
                q = 0.0
                for ns_idx, prob in enumerate(env.P[s_idx, a_idx]):
                    if prob > 0:
                        ns = env.states[ns_idx]
                        q += prob * (env.R[s_idx, a_idx] + env.gamma * V[ns])
                best_value = max(best_value, q)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
        if delta < theta:
            print(f"Value Iteration converged in {iteration+1} iterations (delta={delta:.6f})")
            break
    
    # Extract optimal policy
    policy = {}
    for s in env.states:
        if s == env.goal or s in env.holes:
            policy[s] = 'up'
        else:
            s_idx = env.state_to_idx[s]
            best_value = -np.inf
            best_action = env.actions[0]
            for a_idx, action in enumerate(env.actions):
                q = 0.0
                for ns_idx, prob in enumerate(env.P[s_idx, a_idx]):
                    if prob > 0:
                        ns = env.states[ns_idx]
                        q += prob * (env.R[s_idx, a_idx] + env.gamma * V[ns])
                if q > best_value:
                    best_value = q
                    best_action = action
            policy[s] = best_action
    return policy, V


# ------------------------------------------------------------
# 6. Monte Carlo Policy Evaluation (Model-Free)
# ------------------------------------------------------------
def monte_carlo_evaluation(env, policy, n_episodes=5000, alpha=0.1):
    """First-visit MC prediction for V^π"""
    V = {s: 0.0 for s in env.states}
    returns = defaultdict(list)
    
    for ep in range(n_episodes):
        # Generate episode
        state = (0, 0)  # Start at top-left
        episode = []
        while True:
            action = policy[state]
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        
        # First-visit MC update
        G = 0.0
        visited = set()
        for state, action, reward in reversed(episode):
            G = env.gamma * G + reward
            if state not in visited:
                visited.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    
    return V


# ------------------------------------------------------------
# 7. Temporal Difference Learning (TD(0))
# ------------------------------------------------------------
def td_zero_evaluation(env, policy, n_episodes=2000, alpha=0.1):
    """TD(0) prediction for V^π"""
    V = {s: 0.0 for s in env.states}
    
    for ep in range(n_episodes):
        state = (0, 0)
        while True:
            action = policy[state]
            next_state, reward, done = env.step(state, action)
            # TD(0) update
            V[state] += alpha * (reward + env.gamma * V[next_state] - V[state])
            state = next_state
            if done:
                break
    return V


# ------------------------------------------------------------
# 8. Q-Learning (Model-Free Control)
# ------------------------------------------------------------
def q_learning(env, n_episodes=5000, alpha=0.1, epsilon=0.1):
    """Off-policy TD control: Q-learning"""
    Q = {(s, a): 0.0 for s in env.states for a in env.actions}
    
    for ep in range(n_episodes):
        state = (0, 0)
        while True:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                q_vals = [Q[(state, a)] for a in env.actions]
                action = env.actions[np.argmax(q_vals)]
            
            next_state, reward, done = env.step(state, action)
            
            # Q-learning update
            max_next_q = max(Q[(next_state, a)] for a in env.actions)
            Q[(state, action)] += alpha * (reward + env.gamma * max_next_q - Q[(state, action)])
            
            state = next_state
            if done:
                break
    
    # Extract policy
    policy = {}
    for s in env.states:
        if s == env.goal or s in env.holes:
            policy[s] = 'up'
        else:
            q_vals = [Q[(s, a)] for a in env.actions]
            policy[s] = env.actions[np.argmax(q_vals)]
    return policy, Q


# ------------------------------------------------------------
# 9. Experiment Runner & Analysis
# ------------------------------------------------------------
def run_experiment():
    print("=" * 60)
    print("DAY 36: RL FOUNDATIONS - MDP, REWARDS, POLICY")
    print("=" * 60)
    
    # Create environment
    env = GridWorld(size=4, goal=(3, 3), holes=[(1, 1), (2, 2)], gamma=0.9)
    print(f"\nEnvironment: {env.size}x{env.size} Grid World")
    print(f"Goal: {env.goal}, Holes: {env.holes}")
    print(f"Discount factor γ = {env.gamma}")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    
    # --------------------------------------------------------
    # Experiment 1: Policy Iteration (Model-based)
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: POLICY ITERATION (Model-Based)")
    print("=" * 60)
    pi_policy, pi_V = policy_iteration(env)
    
    # --------------------------------------------------------
    # Experiment 2: Value Iteration (Model-Based)
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: VALUE ITERATION (Model-Based)")
    print("=" * 60)
    vi_policy, vi_V = value_iteration(env)
    env.render_policy(vi_policy)
    env.render_values(vi_V)
    
    # Verify they match
    match = all(pi_policy[s] == vi_policy[s] for s in env.states)
    print(f"Policies match: {match}")
    
    # --------------------------------------------------------
    # Experiment 3: Model-Free Prediction (MC vs TD)
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: MODEL-FREE PREDICTION (MC vs TD)")
    print("=" * 60)
    
    # Use the optimal policy for evaluation
    optimal_policy = vi_policy
    
    print("\nMonte Carlo Evaluation (5000 episodes)...")
    mc_V = monte_carlo_evaluation(env, optimal_policy, n_episodes=5000)
    env.render_values(mc_V)
    
    print("\nTD(0) Evaluation (2000 episodes)...")
    td_V = td_zero_evaluation(env, optimal_policy, n_episodes=2000)
    env.render_values(td_V)
    
    # Compare with true values
    print("\nComparison with True Values (VI):")
    for s in env.states:
        if s != env.goal and s not in env.holes:
            print(f"  {s}: VI={vi_V[s]:6.2f}, MC={mc_V[s]:6.2f}, TD={td_V[s]:6.2f}")
    
    # --------------------------------------------------------
    # Experiment 4: Model-Free Control (Q-Learning)
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Q-LEARNING (Model-Free Control)")
    print("=" * 60)
    ql_policy, Q = q_learning(env, n_episodes=10000, alpha=0.1, epsilon=0.1)
    env.render_policy(ql_policy)
    
    # Compare with optimal
    match = all(ql_policy[s] == vi_policy[s] for s in env.states if s != env.goal and s not in env.holes)
    print(f"Q-Learning policy matches optimal: {match}")
    
    # --------------------------------------------------------
    # Experiment 5: Reward Shaping Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: REWARD SHAPING EFFECTS")
    print("=" * 60)
    
    # Compare different step costs
    for step_cost in [-0.01, -0.1, -0.5, -1.0]:
        test_env = GridWorld(size=4, goal=(3, 3), holes=[(1, 1), (2, 2)], gamma=0.9)
        # Override rewards
        for s_idx, (r, c) in enumerate(test_env.states):
            if (r, c) != test_env.goal and (r, c) not in test_env.holes:
                test_env.R[s_idx, :] = step_cost
        
        policy, V = value_iteration(test_env, theta=1e-4)
        path_length = simulate_path(test_env, policy)
        print(f"  Step cost {step_cost:5.2f}: Path length = {path_length}, V(start)={V[(0,0)]:6.2f}")
    
    # --------------------------------------------------------
    # Experiment 6: Discount Factor Sensitivity
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: DISCOUNT FACTOR γ SENSITIVITY")
    print("=" * 60)
    
    for gamma in [0.5, 0.7, 0.9, 0.95, 0.99]:
        test_env = GridWorld(size=4, goal=(3, 3), holes=[(1, 1), (2, 2)], gamma=gamma)
        policy, V = value_iteration(test_env, theta=1e-4)
        print(f"  γ = {gamma:.2f}: V(start) = {V[(0,0)]:6.2f}, Policy at (0,0) = {policy[(0,0)]}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


def simulate_path(env, policy, max_steps=20):
    """Simulate a trajectory following policy"""
    state = (0, 0)
    steps = 0
    while state != env.goal and state not in env.holes and steps < max_steps:
        action = policy[state]
        # Use most likely transition
        s_idx = env.state_to_idx[state]
        a_idx = env.action_idx[action]
        ns_idx = np.argmax(env.P[s_idx, a_idx])
        state = env.states[ns_idx]
        steps += 1
    return steps if state == env.goal else -1


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    run_experiment()