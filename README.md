# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  53 / ∞   Topics covered: 12/41
[████████░░░░░░░░░░░░░░░░░░░░░░] 29%
🔥 Current streak: 6 days
```


## 📅 Latest Entry

**Day 53 — Optimizers: SGD, Momentum, RMSProp, Adam, AdamW**
🗓️ August 19, 2026
📖 [Read entry →](journal/entries/day_053_optimizers:-sgd,-momentum,-rmsprop,-adam.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 053 | [Optimizers Sgd Momentum Rmsprop Adam Adamw](journal/entries/day_053_optimizers-sgd-momentum-rmsprop-adam-adamw.md) |
| 052 | [Vanishing And Exploding Gradients In Deep Anns](journal/entries/day_052_vanishing-and-exploding-gradients-in-deep-anns.md) |
| 051 | [Universal Approximation Theorem](journal/entries/day_051_universal-approximation-theorem.md) |
| 050 | [Feedforward Neural Networks Ann Architecture](journal/entries/day_050_feedforward-neural-networks-ann-architecture.md) |
| 049 | [Learning Rate Schedules Step Decay Cosine Anneal](journal/entries/day_049_learning-rate-schedules-step-decay-cosine-anneal.md) |

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
