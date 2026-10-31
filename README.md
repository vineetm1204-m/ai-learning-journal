# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day 102 / ∞   Topics covered: 20/41
[██████████████░░░░░░░░░░░░░░░░] 48%
🔥 Current streak: 14 days
```


## 📅 Latest Entry

**Day 102 — Data augmentation for image tasks**
🗓️ October 31, 2026
📖 [Read entry →](journal/entries/day_102_data-augmentation-for-image-tasks.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 102 | [Data Augmentation For Image Tasks](journal/entries/day_102_data-augmentation-for-image-tasks.md) |
| 101 | [Transfer Learning And Fine Tuning With Cnns](journal/entries/day_101_transfer-learning-and-fine-tuning-with-cnns.md) |
| 100 | [Resnets And Skip Connections](journal/entries/day_100_resnets-and-skip-connections.md) |
| 099 | [Classic Architectures Lenet Alexnet Vgg](journal/entries/day_099_classic-architectures-lenet-alexnet-vgg.md) |
| 098 | [Cnn Receptive Field And Spatial Hierarchy](journal/entries/day_098_cnn-receptive-field-and-spatial-hierarchy.md) |

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
