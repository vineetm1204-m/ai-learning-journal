# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  86 / ∞   Topics covered: 4/41
[██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 9%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 86 — Gradient Descent: batch, mini-batch, stochastic**
🗓️ October 12, 2026
📖 [Read entry →](journal/entries/day_086_gradient-descent:-batch,-mini-batch,-sto.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 086 | [Gradient Descent Batch Mini Batch Stochastic](journal/entries/day_086_gradient-descent-batch-mini-batch-stochastic.md) |
| 085 | [Loss Functions Mse Cross Entropy Huber Loss](journal/entries/day_085_loss-functions-mse-cross-entropy-huber-loss.md) |
| 084 | [Activation Functions Sigmoid Tanh Relu Leaky R](journal/entries/day_084_activation-functions-sigmoid-tanh-relu-leaky-r.md) |
| 083 | [Perceptrons And The Biological Neuron Analogy](journal/entries/day_083_perceptrons-and-the-biological-neuron-analogy.md) |
| 082 | [Federated Learning And Privacy Preserving Ml](journal/entries/day_082_federated-learning-and-privacy-preserving-ml.md) |

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
