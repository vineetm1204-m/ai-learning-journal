# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  31 / ∞   Topics covered: 31/41
[██████████████████████░░░░░░░░] 75%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 31 — GPT and autoregressive language modeling**
🗓️ July 26, 2026
📖 [Read entry →](journal/entries/day_031_gpt-and-autoregressive-language-modeling.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 031 | [Gpt And Autoregressive Language Modeling](journal/entries/day_031_gpt-and-autoregressive-language-modeling.md) |
| 030 | [Bert And Masked Language Modeling](journal/entries/day_030_bert-and-masked-language-modeling.md) |
| 029 | [Positional Encoding In Transformers](journal/entries/day_029_positional-encoding-in-transformers.md) |
| 028 | [The Transformer Architecture Attention Is All You](journal/entries/day_028_the-transformer-architecture-attention-is-all-you.md) |
| 027 | [Attention Mechanisms Self Attention And Cross Att](journal/entries/day_027_attention-mechanisms-self-attention-and-cross-att.md) |

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
