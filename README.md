# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  22 / ∞   Topics covered: 22/41
[███████████████░░░░░░░░░░░░░░░] 53%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 22 — Recurrent Neural Networks (RNN) unrolled through time**
🗓️ July 14, 2026
📖 [Read entry →](journal/entries/day_022_recurrent-neural-networks-(rnn)-unrolled.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 022 | [Recurrent Neural Networks Rnn Unrolled Through T](journal/entries/day_022_recurrent-neural-networks-rnn-unrolled-through-t.md) |
| 021 | [Object Detection Yolo R Cnn Family Overview](journal/entries/day_021_object-detection-yolo-r-cnn-family-overview.md) |
| 020 | [Data Augmentation For Image Tasks](journal/entries/day_020_data-augmentation-for-image-tasks.md) |
| 019 | [Transfer Learning And Fine Tuning With Cnns](journal/entries/day_019_transfer-learning-and-fine-tuning-with-cnns.md) |
| 018 | [Resnets And Skip Connections](journal/entries/day_018_resnets-and-skip-connections.md) |

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
