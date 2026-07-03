importnumpy as np
import matplotlib.pyplot as plt
import time
import json
from itertools import product
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# Synthetic Dataset
# ============================================================
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# Search Space Definition
# ============================================================
param_space = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (256, 128)],
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate_init': [1e-3, 1e-2, 1e-1],
    'batch_size': [32, 64, 128],
    'max_iter': [200]
}

def create_model(params):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            batch_size=params['batch_size'],
            max_iter=params['max_iter'],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ))
    ])

def evaluate_params(params, cv=3):
    model = create_model(params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return np.mean(scores), np.std(scores)

# ============================================================
# Strategy 1: Grid Search (Exhaustive)
# ============================================================
def grid_search(param_space, max_combos=50):
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]
    all_combos = list(product(*values))
    
    if len(all_combos) > max_combos:
        idx = np.random.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    
    results = []
    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        mean_score, std_score = evaluate_params(params)
        results.append({**params, 'mean_cv_score': mean_score, 'std_cv_score': std_score})
        print(f"  Grid [{i+1}/{len(all_combos)}] Score: {mean_score:.4f} ± {std_score:.4f}")
    return results

# ============================================================
# Strategy 2: Random Search
# ============================================================
def random_search(param_space, n_iter=50):
    results = []
    for i in range(n_iter):
        params = {k: np.random.choice(v) for k, v in param_space.items()}
        mean_score, std_score = evaluate_params(params)
        results.append({**params, 'mean_cv_score': mean_score, 'std_cv_score': std_score})
        print(f"  Random [{i+1}/{n_iter}] Score: {mean_score:.4f} ± {std_score:.4f}")
    return results

# ============================================================
# Strategy 3: Bayesian Optimization (Simple GP-based)
# ============================================================
class SimpleBayesianOptimizer:
    def __init__(self, param_space, n_initial=5):
        self.param_space = param_space
        self.keys = list(param_space.keys())
        self.categorical_dims = {k: len(v) for k, v in param_space.items()}
        self.X_obs = []
        self.y_obs = []
        self.n_initial = n_initial
        
    def _encode(self, params):
        return [self.param_space[k].index(params[k]) for k in self.keys]
    
    def _decode(self, encoded):
        return {k: self.param_space[k][int(round(v))] for k, v in zip(self.keys, encoded)}
    
    def _rbf_kernel(self, x1, x2, length_scale=1.0):
        x1 = np.array(x1)
        x2 = np.array(x2)
        dist = np.sum((x1 - x2) ** 2)
        return np.exp(-dist / (2 * length_scale ** 2))
    
    def _predict(self, x_test):
        if len(self.X_obs) == 0:
            return 0.0, 1.0
        
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        K = np.array([[self._rbf_kernel(xi, xj) for xj in X] for xi in X]) + 1e-6 * np.eye(len(X))
        k_star = np.array([self._rbf_kernel(x_test, xi) for xi in X])
        
        try:
            alpha = np.linalg.solve(K, y)
            mu = k_star @ alpha
            v = np.linalg.solve(K, k_star)
            var = self._rbf_kernel(x_test, x_test) - k_star @ v
            return mu, max(var, 1e-6)
        except:
            return np.mean(y), np.var(y)
    
    def _expected_improvement(self, x, xi=0.01):
        mu, var = self._predict(x)
        sigma = np.sqrt(var)
        if sigma == 0:
            return 0.0
        best = max(self.y_obs) if self.y_obs else 0
        z = (mu - best - xi) / sigma
        from scipy.stats import norm
        ei = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return max(ei, 0)
    
    def suggest(self):
        if len(self.X_obs) < self.n_initial:
            return {k: np.random.choice(v) for k, v in self.param_space.items()}
        
        best_ei = -1
        best_x = None
        for _ in range(200):
            x_cand = [np.random.uniform(0, self.categorical_dims[k]-1) for k in self.keys]
            ei = self._expected_improvement(x_cand)
            if ei > best_ei:
                best_ei = ei
                best_x = x_cand
        return self._decode(best_x)
    
    def update(self, params, score):
        self.X_obs.append(self._encode(params))
        self.y_obs.append(score)

def bayesian_optimization(param_space, n_iter=30, n_initial=5):
    opt = SimpleBayesianOptimizer(param_space, n_initial)
    results = []
    for i in range(n_iter):
        params = opt.suggest()
        mean_score, std_score = evaluate_params(params)
        opt.update(params, mean_score)
        results.append({**params, 'mean_cv_score': mean_score, 'std_cv_score': std_score})
        print(f"  Bayes [{i+1}/{n_iter}] Score: {mean_score:.4f} ± {std_score:.4f} (EI-based)")
    return results

# ============================================================
# Strategy 4: Successive Halving (Resource Allocation)
# ============================================================
def successive_halving(param_space, n_configs=27, min_resource=50, max_resource=200, reduction_factor=3):
    configs = [{k: np.random.choice(v) for k, v in param_space.items()} for _ in range(n_configs)]
    resources = min_resource
    round_num = 0
    
    all_results = []
    
    while len(configs) > 1 and resources <= max_resource:
        round_num += 1
        print(f"  SH Round {round_num}: {len(configs)} configs, resource={resources}")
        
        round_results = []
        for i, params in enumerate(configs):
            model = create_model(params)
            model.set_params(mlp__max_iter=resources)
            scores = cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy', n_jobs=-1)
            mean_score = np.mean(scores)
            round_results.append((mean_score, params))
            all_results.append({**params, 'mean_cv_score': mean_score, 'resource': resources, 'round': round_num})
            print(f"    Config {i+1}: {mean_score:.4f}")
        
        round_results.sort(key=lambda x: x[0], reverse=True)
        n_keep = max(1, len(configs) // reduction_factor)
        configs = [p for _, p in round_results[:n_keep]]
        resources *= reduction_factor
    
    return all_results

# ============================================================
# Run All Strategies
# ============================================================
print("=" * 60)
print("HYPERPARAMETER TUNING STRATEGIES COMPARISON")
print("=" * 60)

strategies = {}
n_trials = 30

print("\n[1/4] GRID SEARCH (subset)")
start = time.time()
strategies['Grid Search'] = grid_search(param_space, max_combos=n_trials)
print(f"  Time: {time.time()-start:.1f}s")

print("\n[2/4] RANDOM SEARCH")
start = time.time()
strategies['Random Search'] = random_search(param_space, n_iter=n_trials)
print(f"  Time: {time.time()-start:.1f}s")

print("\n[3/4] BAYESIAN OPTIMIZATION")
start = time.time()
strategies['Bayesian Opt'] = bayesian_optimization(param_space, n_iter=n_trials)
print(f"  Time: {time.time()-start:.1f}s")

print("\n[4/4] SUCCESSIVE HALVING")
start = time.time()
strategies['Successive Halving'] = successive_halving(param_space, n_configs=27)
print(f"  Time: {time.time()-start:.1f}s")

# ============================================================
# Analysis & Visualization
# ============================================================
def get_best_result(results):
    if not results:
        return None
    return max(results, key=lambda x: x['mean_cv_score'])

print("\n" + "=" * 60)
print("SUMMARY: BEST CONFIGURATION PER STRATEGY")
print("=" * 60)
for name, results in strategies.items():
    best = get_best_result(results)
    if best:
        print(f"\n{name}:")
        print(f"  CV Score: {best['mean_cv_score']:.4f} ± {best.get('std_cv_score', 0):.4f}")
        for k, v in best.items():
            if k not in ['mean_cv_score', 'std_cv_score', 'resource', 'round']:
                print(f"  {k}: {v}")

# Convergence plot
plt.figure(figsize=(12, 8))

# Plot 1: Convergence curves
plt.subplot(2, 2, 1)
for name, results in strategies.items():
    if name == 'Successive Halving':
        continue
    scores = [r['mean_cv_score'] for r in results]
    best_so_far = np.maximum.accumulate(scores)
    plt.plot(range(1, len(best_so_far)+1), best_so_far, label=name, marker='o', markersize=3)
plt.xlabel('Trial')
plt.ylabel('Best CV Accuracy')
plt.title('Convergence: Best Score Over Trials')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Score distributions
plt.subplot(2, 2, 2)
data = []
labels = []
for name, results in strategies.items():
    if name == 'Successive Halving':
        continue
    scores = [r['mean_cv_score'] for r in results]
    data.append(scores)
    labels.append(name)
plt.boxplot(data, labels=labels)
plt.ylabel('CV Accuracy')
plt.title('Score Distribution Across Trials')
plt.grid(True, alpha=0.3)

# Plot 3: Successive Halving progression
plt.subplot(2, 2, 3)
sh_results = strategies['Successive Halving']
rounds = sorted(set(r['round'] for r in sh_results))
for r in rounds:
    round_scores = [res['mean_cv_score'] for res in sh_results if res['round'] == r]
    plt.scatter([r]*len(round_scores), round_scores, alpha=0.6, label=f'Round {r}' if r==rounds[0] else "")
plt.xlabel('Round')
plt.ylabel('CV Accuracy')
plt.title('Successive Halving: Scores by Round')
plt.grid(True, alpha=0.3)

# Plot 4: Parameter importance (Random Search)
plt.subplot(2, 2, 4)
rs_results = strategies['Random Search']
param_names = [k for k in param_space.keys() if k != 'max_iter']
importances = {}
for p in param_names:
    values = [r[p] for r in rs_results]
    scores = [r['mean_cv_score'] for r in rs_results]
    unique_vals = sorted(set(str(v) for v in values))
    if len(unique_vals) > 1:
        group_means = []
        for uv in unique_vals:
            mask = [str(v)==uv for v in values]
            group_means.append(np.mean([s for s, m in zip(scores, mask) if m]))
        importances[p] = np.std(group_means)

if importances:
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    plt.barh([p for p,_ in sorted_params], [v for _,v in sorted_params])
    plt.xlabel('Score Std Across Values (Proxy Importance)')
    plt.title('Parameter Sensitivity (Random Search)')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('hyperparameter_tuning_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'hyperparameter_tuning_comparison.png'")

# Final test evaluation
print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION (Best Overall Config)")
print("=" * 60)
all_results = []
for results in strategies.values():
    all_results.extend(results)
overall_best = get_best_result(all_results)

if overall_best:
    print(f"\nBest Config: {overall_best}")
    final_model = create_model(overall_best)
    final_model.fit(X_train, y_train)
    test_acc = final_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

# Save results
with open('tuning_results.json', 'w') as f:
    serializable = {}
    for k, v in strategies.items():
        serializable[k] = []
        for r in v:
            clean = {}
            for kk, vv in r.items():
                if isinstance(vv, (np.integer, np.floating)):
                    clean[kk] = float(vv)
                elif isinstance(vv, tuple):
                    clean[kk] = str(vv)
                else:
                    clean[kk] = vv
            serializable[k].append(clean)
    json.dump(serializable, f, indent=2)
print("\nResults saved to 'tuning_results.json'")