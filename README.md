# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day 113 / ∞   Topics covered: 31/41
[██████████████████████░░░░░░░░] 75%
🔥 Current streak: 2 days
```


## 📅 Latest Entry

**Day 113 — GPT and autoregressive language modeling**
🗓️ November 13, 2026
📖 [Read entry →](journal/entries/day_113_gpt-and-autoregressive-language-modeling.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 113 | [Gpt And Autoregressive Language Modeling](journal/entries/day_113_gpt-and-autoregressive-language-modeling.md) |
| 112 | [Bert And Masked Language Modeling](journal/entries/day_112_bert-and-masked-language-modeling.md) |
| 111 | [Positional Encoding In Transformers](journal/entries/day_111_positional-encoding-in-transformers.md) |
| 110 | [The Transformer Architecture Attention Is All You](journal/entries/day_110_the-transformer-architecture-attention-is-all-you.md) |
| 109 | [Attention Mechanisms Self Attention And Cross Att](journal/entries/day_109_attention-mechanisms-self-attention-and-cross-att.md) |

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
