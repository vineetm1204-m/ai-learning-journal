# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  33 / ∞   Topics covered: 33/41
[████████████████████████░░░░░░] 80%
🔥 Current streak: 3 days
```


## 📅 Latest Entry

**Day 33 — Variational Autoencoders (VAEs)**
🗓️ July 28, 2026
📖 [Read entry →](journal/entries/day_033_variational-autoencoders-(vaes).md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 033 | [Variational Autoencoders Vaes](journal/entries/day_033_variational-autoencoders-vaes.md) |
| 032 | [Generative Adversarial Networks Gans Generator](journal/entries/day_032_generative-adversarial-networks-gans-generator.md) |
| 031 | [Gpt And Autoregressive Language Modeling](journal/entries/day_031_gpt-and-autoregressive-language-modeling.md) |
| 030 | [Bert And Masked Language Modeling](journal/entries/day_030_bert-and-masked-language-modeling.md) |
| 029 | [Positional Encoding In Transformers](journal/entries/day_029_positional-encoding-in-transformers.md) |

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
