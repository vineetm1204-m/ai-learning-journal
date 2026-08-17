# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  51 / ∞   Topics covered: 10/41
[███████░░░░░░░░░░░░░░░░░░░░░░░] 24%
🔥 Current streak: 4 days
```


## 📅 Latest Entry

**Day 51 — Universal approximation theorem**
🗓️ August 17, 2026
📖 [Read entry →](journal/entries/day_051_universal-approximation-theorem.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 051 | [Universal Approximation Theorem](journal/entries/day_051_universal-approximation-theorem.md) |
| 050 | [Feedforward Neural Networks Ann Architecture](journal/entries/day_050_feedforward-neural-networks-ann-architecture.md) |
| 049 | [Learning Rate Schedules Step Decay Cosine Anneal](journal/entries/day_049_learning-rate-schedules-step-decay-cosine-anneal.md) |
| 048 | [Regularization L1 L2 Dropout Batchnorm](journal/entries/day_048_regularization-l1-l2-dropout-batchnorm.md) |
| 047 | [Weight Initialization Strategies Xavier He Rand](journal/entries/day_047_weight-initialization-strategies-xavier-he-rand.md) |

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
