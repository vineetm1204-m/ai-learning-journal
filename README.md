# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  25 / ∞   Topics covered: 25/41
[██████████████████░░░░░░░░░░░░] 60%
🔥 Current streak: 4 days
```


## 📅 Latest Entry

**Day 25 — Gated Recurrent Units (GRU)**
🗓️ July 17, 2026
📖 [Read entry →](journal/entries/day_025_gated-recurrent-units-(gru).md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 025 | [Gated Recurrent Units Gru](journal/entries/day_025_gated-recurrent-units-gru.md) |
| 024 | [Long Short Term Memory Lstm Gates Explained](journal/entries/day_024_long-short-term-memory-lstm-gates-explained.md) |
| 023 | [Bptt Backpropagation Through Time](journal/entries/day_023_bptt-backpropagation-through-time.md) |
| 022 | [Recurrent Neural Networks Rnn Unrolled Through T](journal/entries/day_022_recurrent-neural-networks-rnn-unrolled-through-t.md) |
| 021 | [Object Detection Yolo R Cnn Family Overview](journal/entries/day_021_object-detection-yolo-r-cnn-family-overview.md) |

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
