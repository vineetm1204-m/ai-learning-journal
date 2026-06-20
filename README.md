# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day   6 / ∞   Topics covered: 6/41
[████░░░░░░░░░░░░░░░░░░░░░░░░░░] 14%
🔥 Current streak: 4 days
```


## 📅 Latest Entry

**Day 6 — Weight initialization strategies: Xavier, He, random**
🗓️ June 20, 2026
📖 [Read entry →](journal/entries/day_006_weight-initialization-strategies:-xavier.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 006 | [Weight Initialization Strategies Xavier He Rand](journal/entries/day_006_weight-initialization-strategies-xavier-he-rand.md) |
| 005 | [Backpropagation Intuition And The Chain Rule](journal/entries/day_005_backpropagation-intuition-and-the-chain-rule.md) |
| 004 | [Gradient Descent Batch Mini Batch Stochastic](journal/entries/day_004_gradient-descent-batch-mini-batch-stochastic.md) |
| 003 | [Loss Functions Mse Cross Entropy Huber Loss](journal/entries/day_003_loss-functions-mse-cross-entropy-huber-loss.md) |
| 002 | [Activation Functions Sigmoid Tanh Relu Leaky R](journal/entries/day_002_activation-functions-sigmoid-tanh-relu-leaky-r.md) |

[Browse all entries →](journal/entries/)


## 🗂️ Curriculum Overview

Topics span **Foundations → ANN → CNN → RNN → Transformers → Advanced**.
Each entry contains:

| Section | Details |
|---------|---------|
| 🧠 Concept | 200-word intuitive explanation |
| ✅ Key Points | 5 crisp bullets |
| 🏗️ Diagram | ASCII diagram or formula |
| 💡 Analogy | Real-world comparison |
| ⚠️ Mistakes | 3 common beginner traps |
| 🧪 Experiment | Runnable Python (PyTorch/NumPy) |
| 📚 Resources | 2 free links |
| 🧩 Quiz | 2 MCQs with hidden answers |


## 🚀 Run Experiments Locally

```bash
git clone https://github.com/vineetm1204-m/ai-learning-journal.git
cd ai-learning-journal
pip install torch numpy openai
python journal/experiments/day_001_*.py
```


## ⚙️ How It Works

```
GitHub Actions (cron: daily 6 AM UTC)
        │
        ▼
generate_entry.py
        │
        ├── OpenRouter → concept notes (JSON)
        ├── OpenRouter → experiment code (.py)
        ├── Builds Markdown entry
        ├── Updates README.md
        └── git commit & push
```


*Built with ❤️ by Vineet Mittal*
