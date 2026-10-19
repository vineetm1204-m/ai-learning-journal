# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  90 / ∞   Topics covered: 8/41
[█████░░░░░░░░░░░░░░░░░░░░░░░░░] 19%
🔥 Current streak: 2 days
```


## 📅 Latest Entry

**Day 90 — Learning rate schedules: step decay, cosine annealing, warm restarts**
🗓️ October 19, 2026
📖 [Read entry →](journal/entries/day_090_learning-rate-schedules:-step-decay,-cos.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 090 | [Learning Rate Schedules Step Decay Cosine Anneal](journal/entries/day_090_learning-rate-schedules-step-decay-cosine-anneal.md) |
| 089 | [Regularization L1 L2 Dropout Batchnorm](journal/entries/day_089_regularization-l1-l2-dropout-batchnorm.md) |
| 088 | [Weight Initialization Strategies Xavier He Rand](journal/entries/day_088_weight-initialization-strategies-xavier-he-rand.md) |
| 087 | [Backpropagation Intuition And The Chain Rule](journal/entries/day_087_backpropagation-intuition-and-the-chain-rule.md) |
| 086 | [Gradient Descent Batch Mini Batch Stochastic](journal/entries/day_086_gradient-descent-batch-mini-batch-stochastic.md) |

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
