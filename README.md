# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with OpenRouter and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day 123 / ∞   Topics covered: 41/41
[██████████████████████████████] 100%
🔥 Current streak: 12 days
```


## 📅 Latest Entry

**Day 123 — Federated learning and privacy-preserving ML**
🗓️ November 23, 2026
📖 [Read entry →](journal/entries/day_123_federated-learning-and-privacy-preservin.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
| 123 | [Federated Learning And Privacy Preserving Ml](journal/entries/day_123_federated-learning-and-privacy-preserving-ml.md) |
| 122 | [Quantization And Model Pruning For Deployment](journal/entries/day_122_quantization-and-model-pruning-for-deployment.md) |
| 121 | [Neural Architecture Search Nas](journal/entries/day_121_neural-architecture-search-nas.md) |
| 120 | [Multi Task And Meta Learning Overview](journal/entries/day_120_multi-task-and-meta-learning-overview.md) |
| 119 | [Deep Q Networks Dqn](journal/entries/day_119_deep-q-networks-dqn.md) |

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
