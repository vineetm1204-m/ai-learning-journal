# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  94 / ∞   Topics covered: 12/41
[████████░░░░░░░░░░░░░░░░░░░░░░] 29%
🔥 Current streak: 6 days
```


## 📅 Latest Entry

**Day 94 — Optimizers: SGD, Momentum, RMSProp, Adam, AdamW**
🗓️ October 23, 2026
📖 [Read entry →](journal/entries/day_094_optimizers:-sgd,-momentum,-rmsprop,-adam.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 094 | [Optimizers Sgd Momentum Rmsprop Adam Adamw](journal/entries/day_094_optimizers-sgd-momentum-rmsprop-adam-adamw.md) |
| 093 | [Vanishing And Exploding Gradients In Deep Anns](journal/entries/day_093_vanishing-and-exploding-gradients-in-deep-anns.md) |
| 092 | [Universal Approximation Theorem](journal/entries/day_092_universal-approximation-theorem.md) |
| 091 | [Feedforward Neural Networks Ann Architecture](journal/entries/day_091_feedforward-neural-networks-ann-architecture.md) |
| 090 | [Learning Rate Schedules Step Decay Cosine Anneal](journal/entries/day_090_learning-rate-schedules-step-decay-cosine-anneal.md) |

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
