# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  77 / ∞   Topics covered: 36/41
[██████████████████████████░░░░] 87%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 77 — Reinforcement Learning foundations: MDP, rewards, policy**
🗓️ September 25, 2026
📖 [Read entry →](journal/entries/day_077_reinforcement-learning-foundations:-mdp,.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 077 | [Reinforcement Learning Foundations Mdp Rewards](journal/entries/day_077_reinforcement-learning-foundations-mdp-rewards.md) |
| 076 | [Graph Neural Networks Gnns Basics](journal/entries/day_076_graph-neural-networks-gnns-basics.md) |
| 075 | [Diffusion Models Intuition](journal/entries/day_075_diffusion-models-intuition.md) |
| 074 | [Variational Autoencoders Vaes](journal/entries/day_074_variational-autoencoders-vaes.md) |
| 073 | [Generative Adversarial Networks Gans Generator](journal/entries/day_073_generative-adversarial-networks-gans-generator.md) |

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
