# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  70 / ∞   Topics covered: 29/41
[█████████████████████░░░░░░░░░] 70%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 70 — Positional encoding in Transformers**
🗓️ September 16, 2026
📖 [Read entry →](journal/entries/day_070_positional-encoding-in-transformers.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 070 | [Positional Encoding In Transformers](journal/entries/day_070_positional-encoding-in-transformers.md) |
| 069 | [The Transformer Architecture Attention Is All You](journal/entries/day_069_the-transformer-architecture-attention-is-all-you.md) |
| 068 | [Attention Mechanisms Self Attention And Cross Att](journal/entries/day_068_attention-mechanisms-self-attention-and-cross-att.md) |
| 067 | [Sequence To Sequence Models And Encoder Decoder](journal/entries/day_067_sequence-to-sequence-models-and-encoder-decoder.md) |
| 066 | [Gated Recurrent Units Gru](journal/entries/day_066_gated-recurrent-units-gru.md) |

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
