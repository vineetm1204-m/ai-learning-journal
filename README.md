# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day 109 / ∞   Topics covered: 27/41
[███████████████████░░░░░░░░░░░] 65%
🔥 Current streak: 2 days
```


## 📅 Latest Entry

**Day 109 — Attention mechanisms: self-attention and cross-attention**
🗓️ November 08, 2026
📖 [Read entry →](journal/entries/day_109_attention-mechanisms:-self-attention-and.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 109 | [Attention Mechanisms Self Attention And Cross Att](journal/entries/day_109_attention-mechanisms-self-attention-and-cross-att.md) |
| 108 | [Sequence To Sequence Models And Encoder Decoder](journal/entries/day_108_sequence-to-sequence-models-and-encoder-decoder.md) |
| 107 | [Gated Recurrent Units Gru](journal/entries/day_107_gated-recurrent-units-gru.md) |
| 106 | [Long Short Term Memory Lstm Gates Explained](journal/entries/day_106_long-short-term-memory-lstm-gates-explained.md) |
| 105 | [Bptt Backpropagation Through Time](journal/entries/day_105_bptt-backpropagation-through-time.md) |

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
