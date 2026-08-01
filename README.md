# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  37 / ∞   Topics covered: 37/41
[███████████████████████████░░░] 90%
🔥 Current streak: 7 days
```


## 📅 Latest Entry

**Day 37 — Deep Q-Networks (DQN)**
🗓️ August 01, 2026
📖 [Read entry →](journal/entries/day_037_deep-q-networks-(dqn).md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 037 | [Deep Q Networks Dqn](journal/entries/day_037_deep-q-networks-dqn.md) |
| 036 | [Reinforcement Learning Foundations Mdp Rewards](journal/entries/day_036_reinforcement-learning-foundations-mdp-rewards.md) |
| 035 | [Graph Neural Networks Gnns Basics](journal/entries/day_035_graph-neural-networks-gnns-basics.md) |
| 034 | [Diffusion Models Intuition](journal/entries/day_034_diffusion-models-intuition.md) |
| 033 | [Variational Autoencoders Vaes](journal/entries/day_033_variational-autoencoders-vaes.md) |

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
