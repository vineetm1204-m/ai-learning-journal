import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys

# ==========================================
# 1. MDP Environment: Deterministic Grid World
# ==========================================
class GridWorld:
    """
    4x4 Grid World.
    State 0 (0,0) is Start. State 15 (3,3) is Goal (Terminal).
    State 5 (1,1) and State 7 (1,3) are Holes (Terminal, negative reward).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal_state = self.n_states - 1
        self.hole_states = [5, 7] # (1,1), (1,3)
        self.terminal_states = [self.goal_state] + self.hole_states
        
        # Precompute transition dynamics P(s' | s, a) and R(s, a, s')
        # For deterministic env, P is a mapping (s, a) -> s'
        self.transitions = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.rewards = np.zeros((self.n_states, self.n_actions))
        
        self._build_dynamics()

    def _build_dynamics(self):
        for s in range(self.n_states):
            r, c = divmod(s, self.size)
            if s in self.terminal_states:
                # Terminal states transition to self with 0 reward
                for a in range(self.n_actions):
                    self.transitions[s, a] = s
                    self.rewards[s, a] = 0.0
                continue

            for a in range(self.n_actions):
                nr, nc = r, c
                if a == 0: nr = max(0, r - 1)      # Up
                elif a == 1: nr = min(self.size - 1, r + 1) # Down
                elif a == 2: nc = max(0, c - 1)      # Left
                elif a == 3: nc = min(self.size - 1, c + 1) # Right
                
                ns = nr * self.size + nc
                self.transitions[s, a] = ns
                
                # Reward Structure
                if ns == self.goal_state:
                    self.rewards[s, a] = 1.0
                elif ns in self.hole_states:
                    self.rewards[s, a] = -1.0
                else:
                    self.rewards[s, a] = -0.04 # Small step cost to encourage speed

    def step(self, state, action):
        """Returns next_state, reward, done"""
        if state in self.terminal_states:
            return state, 0.0, True
        ns = self.transitions[state, action]
        r = self.rewards[state, action]
        done = ns in self.terminal_states
        return ns, r, done

# ==========================================
# 2. Policy Representation & Evaluation
# ==========================================
def policy_evaluation(policy, env, gamma=0.99, theta=1e-6):
    """
    Iterative Policy Evaluation.
    Solves V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) [ R + gamma * V(s') ]
    """
    V = np.zeros(env.n_states)
    while True:
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states: continue
            
            v = 0
            # Policy is deterministic here: policy[s] = action
            a = policy[s]
            ns = env.transitions[s, a]
            r = env.rewards[s, a]
            v = r + gamma * V[ns]
            
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(V, env, gamma=0.99):
    """
    Policy Improvement.
    pi'(s) = argmax_a sum_s' P(s'|s,a) [ R + gamma * V(s') ]
    Returns new_policy, policy_stable (bool)
    """
    new_policy = np.zeros(env.n_states, dtype=int)
    policy_stable = True
    
    for s in range(env.n_states):
        if s in env.terminal_states:
            new_policy[s] = 0 # Arbitrary
            continue
            
        old_action = np.argmax([env.rewards[s, a] + gamma * V[env.transitions[s, a]] for a in range(env.n_actions)])
        # Actually, we need to compare with *current* policy if passed, but here we derive greedy from V
        # Standard PI: compute q for all actions, pick max.
        q_values = np.array([env.rewards[s, a] + gamma * V[env.transitions[s, a]] for a in range(env.n_actions)])
        best_a = np.argmax(q_values)
        new_policy[s] = best_a
        
    return new_policy

def policy_iteration(env, gamma=0.99):
    """Full Policy Iteration Loop."""
    # Random initial policy
    policy = np.random.randint(0, env.n_actions, size=env.n_states)
    for s in env.terminal_states: policy[s] = 0
    
    iteration = 0
    while True:
        iteration += 1
        print(f"  Policy Iteration #{iteration}: Evaluating...")
        V = policy_evaluation(policy, env, gamma)
        print(f"  Policy Iteration #{iteration}: Improving...")
        new_policy = policy_improvement(V, env, gamma)
        
        if np.array_equal(policy, new_policy):
            print(f"  Policy converged at iteration {iteration}.")
            break
        policy = new_policy
    return policy, V

# ==========================================
# 3. Value Iteration (Alternative Foundation)
# ==========================================
def value_iteration(env, gamma=0.99, theta=1e-6):
    """Value Iteration: Combines Evaluation + Improvement in one update."""
    V = np.zeros(env.n_states)
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states: continue
            v = V[s]
            # Bellman Optimality Update
            q_values = [env.rewards[s, a] + gamma * V[env.transitions[s, a]] for a in range(env.n_actions)]
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            print(f"  Value Iteration converged at iteration {iteration}.")
            break
    
    # Extract deterministic policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if s in env.terminal_states: continue
        q_values = [env.rewards[s, a] + gamma * V[env.transitions[s, a]] for a in range(env.n_actions)]
        policy[s] = np.argmax(q_values)
    return policy, V

# ==========================================
# 4. Visualization & Experiment Runner
# ==========================================
ACTION_SYMBOLS = ['↑', '↓', '←', '→']
ACTION_COLORS = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

def print_policy_grid(policy, env, title="Policy"):
    print(f"\n--- {title} ---")
    for r in range(env.size):
        row_str = ""
        for c in range(env.size):
            s = r * env.size + c
            if s == env.goal_state: row_str += " G "
            elif s in env.hole_states: row_str += " H "
            else: row_str += f" {ACTION_SYMBOLS[policy[s]]} "
        print(row_str)

def print_value_grid(V, env, title="Value Function"):
    print(f"\n--- {title} ---")
    for r in range(env.size):
        row_str = ""
        for c in range(env.size):
            s = r * env.size + c
            row_str += f"{V[s]:6.2f} "
        print(row_str)

def plot_results(policy_pi, V_pi, policy_vi, V_vi, env):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cmap = ListedColormap(['white', 'lightblue', 'lightcoral', 'lightgreen', 'gold'])
    
    # Helper to draw grid
    def draw_grid(ax, V, policy, title):
        # Background: Value Function Heatmap
        grid_V = V.reshape(env.size, env.size)
        im = ax.imshow(grid_V, cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotations
        for r in range(env.size):
            for c in range(env.size):
                s = r * env.size + c
                val = V[s]
                if s == env.goal_state:
                    txt = "G\n+1.0"
                    color = 'black'
                elif s in env.hole_states:
                    txt = "H\n-1.0"
                    color = 'white'
                else:
                    txt = f"{ACTION_SYMBOLS[policy[s]]}\n{val:.2f}"
                    color = 'black'
                ax.text(c, r, txt, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=14)
        ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, env.size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    draw_grid(axes[0,0], V_pi, policy_pi, "Policy Iteration Result")
    draw_grid(axes[0,1], V_vi, policy_vi, "Value Iteration Result")
    
    # Difference Heatmap
    diff_V = V_pi - V_vi
    im_diff = axes[1,0].imshow(diff_V.reshape(env.size, env.size), cmap='bwr', interpolation='nearest')
    plt.colorbar(im_diff, ax=axes[1,0], fraction=0.046, pad=0.04)
    axes[1,0].set_title("Difference (PI - VI) Values")
    for r in range(env.size):
        for c in range(env.size):
            axes[1,0].text(c, r, f"{diff_V[r*env.size+c]:.2e}", ha='center', va='center', fontsize=10)
    axes[1,0].set_xticks([]); axes[1,0].set_yticks([])

    # Policy Agreement
    agree = (policy_pi == policy_vi).astype(int)
    agree[env.terminal_states] = 1 # Terminals always agree
    im_agree = axes[1,1].imshow(agree.reshape(env.size, env.size), cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,1].set_title("Policy Agreement (Green=Match)")
    for r in range(env.size):
        for c in range(env.size):
            s = r*env.size+c
            sym = "✓" if agree[s] else "✗"
            axes[1,1].text(c, r, sym, ha='center', va='center', fontsize=20, color='black')
    axes[1,1].set_xticks([]); axes[1,1].set_yticks([])
    
    plt.suptitle("Day 77: RL Foundations - MDP, Rewards, Policy Iteration vs Value Iteration", fontsize=16)
    plt.tight_layout()
    plt.savefig("day77_mdp_foundations.png", dpi=150)
    print("\nPlot saved to 'day77_mdp_foundations.png'")
    # plt.show() # Commented out for headless environments

def run_experiment():
    print("="*60)
    print("DAY 77: REINFORCEMENT LEARNING FOUNDATIONS")
    print("Topic: MDP Formulation, Reward Design, Policy & Value Iteration")
    print("="*60)
    
    # 1. Setup MDP
    env = GridWorld(size=4)
    gamma = 0.95 # Discount factor
    
    print(f"\nEnvironment: {env.size}x{env.size} Grid World")
    print(f"States: {env.n_states} | Actions: {env.n_actions} | Gamma: {gamma}")
    print(f"Terminal States: Goal={env.goal_state}, Holes={env.hole_states}")
    print(f"Reward Structure: Goal=+1, Hole=-1, Step=-0.04")
    
    # 2. Policy Iteration
    print("\n[1/3] Running Policy Iteration...")
    policy_pi, V_pi = policy_iteration(env, gamma)
    print_policy_grid(policy_pi, env, "Optimal Policy (PI)")
    print_value_grid(V_pi, env, "Value Function (PI)")
    
    # 3. Value Iteration
    print("\n[2/3] Running Value Iteration...")
    policy_vi, V_vi = value_iteration(env, gamma)
    print_policy_grid(policy_vi, env, "Optimal Policy (VI)")
    print_value_grid(V_vi, env, "Value Function (VI)")
    
    # 4. Verification & Analysis
    print("\n[3/3] Verification & Analysis...")
    print(f"Policies Match: {np.array_equal(policy_pi, policy_vi)}")
    print(f"Max Value Diff: {np.max(np.abs(V_pi - V_vi)):.6f}")
    
    # Demonstrate Bellman Equation holds for optimal policy
    print("\n--- Bellman Optimality Check (State 0 / Start) ---")
    s = 0
    a_opt = policy_pi[s]
    ns = env.transitions[s, a_opt]
    r = env.rewards[s, a_opt]
    lhs = V_pi[s]
    rhs = r + gamma * V_pi[ns]
    print(f"V*(s={s}) = {lhs:.4f}")
    print(f"R(s,a*) + γV*(s') = {r:.4f} + {gamma}*{V_pi[ns]:.4f} = {rhs:.4f}")
    print(f"Bellman Error: {abs(lhs-rhs):.6f}")
    
    # 5. Visualize
    plot_results(policy_pi, V_pi, policy_vi, V_vi, env)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_experiment()