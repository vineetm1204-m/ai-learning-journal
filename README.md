# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  16 / ∞   Topics covered: 16/41
[███████████░░░░░░░░░░░░░░░░░░░] 39%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 16 — CNN receptive field and spatial hierarchy**
🗓️ July 07, 2026
📖 [Read entry →](journal/entries/day_016_cnn-receptive-field-and-spatial-hierarch.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 016 | [Cnn Receptive Field And Spatial Hierarchy](journal/entries/day_016_cnn-receptive-field-and-spatial-hierarchy.md) |
| 015 | [Pooling Layers Max Pooling Average Pooling](journal/entries/day_015_pooling-layers-max-pooling-average-pooling.md) |
| 014 | [Convolutional Layers Filters Stride Padding](journal/entries/day_014_convolutional-layers-filters-stride-padding.md) |
| 013 | [Hyperparameter Tuning Strategies](journal/entries/day_013_hyperparameter-tuning-strategies.md) |
| 012 | [Optimizers Sgd Momentum Rmsprop Adam Adamw](journal/entries/day_012_optimizers-sgd-momentum-rmsprop-adam-adamw.md) |

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
