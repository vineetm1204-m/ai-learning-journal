# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  40 / ∞   Topics covered: 40/41
[█████████████████████████████░] 97%
🔥 Current streak: 10 days
```


## 📅 Latest Entry

**Day 40 — Quantization and model pruning for deployment**
🗓️ August 04, 2026
📖 [Read entry →](journal/entries/day_040_quantization-and-model-pruning-for-deplo.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 040 | [Quantization And Model Pruning For Deployment](journal/entries/day_040_quantization-and-model-pruning-for-deployment.md) |
| 039 | [Neural Architecture Search Nas](journal/entries/day_039_neural-architecture-search-nas.md) |
| 038 | [Multi Task And Meta Learning Overview](journal/entries/day_038_multi-task-and-meta-learning-overview.md) |
| 037 | [Deep Q Networks Dqn](journal/entries/day_037_deep-q-networks-dqn.md) |
| 036 | [Reinforcement Learning Foundations Mdp Rewards](journal/entries/day_036_reinforcement-learning-foundations-mdp-rewards.md) |

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
