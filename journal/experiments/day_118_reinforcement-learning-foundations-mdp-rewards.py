import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable

# ==========================================
# 1. MDP DEFINITION: Grid World
# ==========================================

class GridWorldMDP:
    """
    A simple 4x4 Grid World MDP.
    State 0 (top-left) is Start. State 15 (bottom-right) is Goal (Terminal).
    State 5 and 7 are Holes (Terminal, negative reward).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    Dynamics: Deterministic transitions (unless hitting wall).
    Rewards: -1 per step, +10 Goal, -10 Hole.
    Gamma: 0.99
    """
    def __init__(self, size: int = 4, gamma: float = 0.99):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma
        
        # Terminal states
        self.goal_state = self.n_states - 1       # 15
        self.hole_states = {5, 7}                 # Indices 5, 7
        self.terminal_states = {self.goal_state} | self.hole_states
        
        # Precompute transition dynamics: P[s][a] = (next_state, reward, done)
        self.P = self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> List[List[Tuple[int, float, bool]]]:
        P = [[None for _ in range(self.n_actions)] for _ in range(self.n_states)]
        
        for s in range(self.n_states):
            row, col = divmod(s, self.size)
            
            if s in self.terminal_states:
                # Terminal states transition to self with 0 reward
                for a in range(self.n_actions):
                    P[s][a] = (s, 0.0, True)
                continue

            for a in range(self.n_actions):
                n_row, n_col = row, col
                if a == 0: n_row = max(0, row - 1)      # Up
                elif a == 1: n_row = min(self.size - 1, row + 1) # Down
                elif a == 2: n_col = max(0, col - 1)    # Left
                elif a == 3: n_col = min(self.size - 1, col + 1) # Right
                
                next_s = n_row * self.size + n_col
                
                if next_s == self.goal_state:
                    reward = 10.0
                    done = True
                elif next_s in self.hole_states:
                    reward = -10.0
                    done = True
                else:
                    reward = -1.0  # Step cost
                    done = False
                    
                P[s][a] = (next_s, reward, done)
        return P

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        return self.P[state][action]

    def render_policy(self, policy: np.ndarray, V: np.ndarray = None):
        """Prints policy arrows and value function to console."""
        arrows = ['^', 'v', '<', '>']
        print("-" * (self.size * 6))
        for r in range(self.size):
            row_str = ""
            val_str = ""
            for c in range(self.size):
                s = r * self.size + c
                if s == self.goal_state:
                    row_str += "  G   "
                    val_str += " 10.0 "
                elif s in self.hole_states:
                    row_str += "  H   "
                    val_str += "-10.0 "
                else:
                    a = policy[s]
                    row_str += f"  {arrows[a]}   "
                    if V is not None:
                        val_str += f"{V[s]:5.1f} "
                    else:
                        val_str += "      "
            print(row_str)
            if V is not None: print(val_str)
        print("-" * (self.size * 6))

# ==========================================
# 2. POLICY EVALUATION (Iterative Policy Evaluation)
# ==========================================

def policy_evaluation(mdp: GridWorldMDP, policy: np.ndarray, theta: float = 1e-6) -> np.ndarray:
    """
    Computes V_pi for a given deterministic policy.
    V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) [ R + gamma * V(s') ]
    Since policy is deterministic: V(s) = R(s, pi(s)) + gamma * V(s')
    """
    V = np.zeros(mdp.n_states)
    while True:
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminal_states: continue
            
            v_old = V[s]
            a = policy[s]
            next_s, reward, done = mdp.step(s, a)
            
            # Bellman Expectation Equation for deterministic policy
            V[s] = reward + (mdp.gamma * V[next_s] if not done else 0.0)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
    return V

# ==========================================
# 3. POLICY IMPROVEMENT & POLICY ITERATION
# ==========================================

def policy_improvement(mdp: GridWorldMDP, V: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Improves policy greedily w.r.t current Value Function.
    Returns new_policy, policy_stable_flag
    """
    policy_stable = True
    new_policy = np.zeros(mdp.n_states, dtype=int)
    
    for s in range(mdp.n_states):
        if s in mdp.terminal_states:
            new_policy[s] = 0 # Arbitrary
            continue
            
        # One-step lookahead to find best action (Action Value Q(s,a))
        action_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            next_s, reward, done = mdp.step(s, a)
            action_values[a] = reward + (mdp.gamma * V[next_s] if not done else 0.0)
            
        best_a = np.argmax(action_values)
        new_policy[s] = best_a
        
        # In a full PI loop, we'd compare with old policy. 
        # Here we just return the greedy policy.
        
    return new_policy, policy_stable

def policy_iteration(mdp: GridWorldMDP) -> Tuple[np.ndarray, np.ndarray]:
    """Full Policy Iteration: Eval -> Improve until convergence."""
    # Random initial policy
    policy = np.random.randint(0, mdp.n_actions, size=mdp.n_states)
    
    print("Starting Policy Iteration...")
    iteration = 0
    while True:
        iteration += 1
        # 1. Policy Evaluation
        V = policy_evaluation(mdp, policy)
        
        # 2. Policy Improvement
        new_policy, _ = policy_improvement(mdp, V)
        
        # Check convergence
        if np.array_equal(policy, new_policy):
            print(f"Policy converged after {iteration} iterations.")
            break
            
        policy = new_policy
        
    return policy, V

# ==========================================
# 4. VALUE ITERATION (Alternative Foundation)
# ==========================================

def value_iteration(mdp: GridWorldMDP, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Optimal V* directly via Bellman Optimality Equation.
    Then extracts deterministic optimal policy.
    """
    V = np.zeros(mdp.n_states)
    print("Starting Value Iteration...")
    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminal_states: continue
            
            v_old = V[s]
            # Bellman Optimality Backup: V(s) = max_a [ R + gamma * V(s') ]
            action_values = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                next_s, reward, done = mdp.step(s, a)
                action_values[a] = reward + (mdp.gamma * V[next_s] if not done else 0.0)
            
            V[s] = np.max(action_values)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            print(f"Value Iteration converged after {iteration} iterations.")
            break
            
    # Extract Policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        if s in mdp.terminal_states: continue
        action_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            next_s, reward, done = mdp.step(s, a)
            action_values[a] = reward + (mdp.gamma * V[next_s] if not done else 0.0)
        policy[s] = np.argmax(action_values)
        
    return policy, V

# ==========================================
# 5. VISUALIZATION
# ==========================================

def plot_results(mdp: GridWorldMDP, V: np.ndarray, policy: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=14)
    
    # 1. Value Function Heatmap
    V_grid = V.reshape((mdp.size, mdp.size))
    im = axes[0].imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title("State Value Function V(s)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Annotate values
    for r in range(mdp.size):
        for c in range(mdp.size):
            s = r * mdp.size + c
            color = 'white' if abs(V[s]) > 5 else 'black'
            if s == mdp.goal_state:
                axes[0].text(c, r, "G", ha="center", va="center", color="blue", fontsize=14, fontweight='bold')
            elif s in mdp.hole_states:
                axes[0].text(c, r, "H", ha="center", va="center", color="red", fontsize=14, fontweight='bold')
            else:
                axes[0].text(c, r, f"{V[s]:.1f}", ha="center", va="center", color=color, fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    
    # 2. Policy Arrows
    axes[1].set_title("Optimal Policy $\pi^*(s)$")
    axes[1].set_xlim(-0.5, mdp.size - 0.5)
    axes[1].set_ylim(mdp.size - 0.5, -0.5) # Invert Y for matrix coords
    axes[1].set_xticks(np.arange(-0.5, mdp.size, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, mdp.size, 1), minor=True)
    axes[1].grid(which="minor", color="black", linestyle='-', linewidth=1)
    axes[1].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    
    arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)} # dx, dy
    
    for r in range(mdp.size):
        for c in range(mdp.size):
            s = r * mdp.size + c
            cx, cy = c, r
            if s == mdp.goal_state:
                axes[1].text(cx, cy, "★", ha="center", va="center", color="gold", fontsize=20, fontweight='bold')
            elif s in mdp.hole_states:
                axes[1].text(cx, cy, "✕", ha="center", va="center", color="red", fontsize=20, fontweight='bold')
            else:
                a = policy[s]
                dx, dy = arrows[a]
                axes[1].arrow(cx, cy, dx, dy, head_width=0.15, head_length=0.15, fc='blue', ec='blue', length_includes_head=True)
                
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. MAIN EXPERIMENT RUNNER
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("DAY 118: RL FOUNDATIONS - MDP, REWARDS, POLICY")
    print("="*60)
    
    # Setup MDP
    mdp = GridWorldMDP(size=4, gamma=0.99)
    print(f"MDP Created: {mdp.size}x{mdp.size} Grid | States: {mdp.n_states} | Actions: {mdp.n_actions}")
    print(f"Terminal States: Goal={mdp.goal_state}, Holes={mdp.hole_states}")
    print(f"Discount Factor (gamma): {mdp.gamma}")
    print("-" * 60)
    
    # --- Experiment 1: Policy Iteration ---
    print("\n[EXPERIMENT 1] POLICY ITERATION")
    pi_policy, pi_V = policy_iteration(mdp)
    mdp.render_policy(pi_policy, pi_V)
    
    # --- Experiment 2: Value Iteration ---
    print("\n[EXPERIMENT 2] VALUE ITERATION")
    vi_policy, vi_V = value_iteration(mdp)
    mdp.render_policy(vi_policy, vi_V)
    
    # --- Verification ---
    print("\n[VERIFICATION] Policies Match?", np.array_equal(pi_policy, vi_policy))
    print(f"Value Functions Close? {np.allclose(pi_V, vi_V)}")
    
    # --- Visualization ---
    try:
        plot_results(mdp, pi_V, pi_policy, "Policy Iteration Results")
        plot_results(mdp, vi_V, vi_policy, "Value Iteration Results")
    except Exception as e:
        print(f"\n[PLOTTING SKIPPED] {e}")
        
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)