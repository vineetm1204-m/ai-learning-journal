# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  19 / ∞   Topics covered: 19/41
[█████████████░░░░░░░░░░░░░░░░░] 46%
🔥 Current streak: 4 days
```


## 📅 Latest Entry

**Day 19 — Transfer learning and fine-tuning with CNNs**
🗓️ July 10, 2026
📖 [Read entry →](journal/entries/day_019_transfer-learning-and-fine-tuning-with-c.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 019 | [Transfer Learning And Fine Tuning With Cnns](journal/entries/day_019_transfer-learning-and-fine-tuning-with-cnns.md) |
| 018 | [Resnets And Skip Connections](journal/entries/day_018_resnets-and-skip-connections.md) |
| 017 | [Classic Architectures Lenet Alexnet Vgg](journal/entries/day_017_classic-architectures-lenet-alexnet-vgg.md) |
| 016 | [Cnn Receptive Field And Spatial Hierarchy](journal/entries/day_016_cnn-receptive-field-and-spatial-hierarchy.md) |
| 015 | [Pooling Layers Max Pooling Average Pooling](journal/entries/day_015_pooling-layers-max-pooling-average-pooling.md) |

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
