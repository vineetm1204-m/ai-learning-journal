# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day  75 / ∞   Topics covered: 34/41
[████████████████████████░░░░░░] 82%
🔥 Current streak: 1 days
```


## 📅 Latest Entry

**Day 75 — Diffusion models intuition**
🗓️ September 22, 2026
📖 [Read entry →](journal/entries/day_075_diffusion-models-intuition.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 075 | [Diffusion Models Intuition](journal/entries/day_075_diffusion-models-intuition.md) |
| 074 | [Variational Autoencoders Vaes](journal/entries/day_074_variational-autoencoders-vaes.md) |
| 073 | [Generative Adversarial Networks Gans Generator](journal/entries/day_073_generative-adversarial-networks-gans-generator.md) |
| 072 | [Gpt And Autoregressive Language Modeling](journal/entries/day_072_gpt-and-autoregressive-language-modeling.md) |
| 071 | [Bert And Masked Language Modeling](journal/entries/day_071_bert-and-masked-language-modeling.md) |

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
