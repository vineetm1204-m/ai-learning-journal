# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  46 / ∞   Topics covered: 5/41
[███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 12%
🔥 Current streak: 4 days
```


## 📅 Latest Entry

**Day 46 — Backpropagation intuition and the chain rule**
🗓️ August 11, 2026
📖 [Read entry →](journal/entries/day_046_backpropagation-intuition-and-the-chain-.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 046 | [Backpropagation Intuition And The Chain Rule](journal/entries/day_046_backpropagation-intuition-and-the-chain-rule.md) |
| 045 | [Gradient Descent Batch Mini Batch Stochastic](journal/entries/day_045_gradient-descent-batch-mini-batch-stochastic.md) |
| 044 | [Loss Functions Mse Cross Entropy Huber Loss](journal/entries/day_044_loss-functions-mse-cross-entropy-huber-loss.md) |
| 043 | [Activation Functions Sigmoid Tanh Relu Leaky R](journal/entries/day_043_activation-functions-sigmoid-tanh-relu-leaky-r.md) |
| 042 | [Perceptrons And The Biological Neuron Analogy](journal/entries/day_042_perceptrons-and-the-biological-neuron-analogy.md) |

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
