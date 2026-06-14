"""
AI Learning Journal - Daily Entry Generator
Uses Gemini API to generate structured Deep Learning journal entries.
"""

import os
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from google import genai
from google.genai import errors as genai_errors

# ── Config ────────────────────────────────────────────────────────────────────

MODEL  = "gemini-2.0-flash"

REPO_ROOT    = Path(__file__).resolve().parent.parent
ENTRIES_DIR  = REPO_ROOT / "journal" / "entries"
EXPERIMENTS_DIR = REPO_ROOT / "journal" / "experiments"
PROGRESS_FILE   = REPO_ROOT / "journal" / "progress.json"
README_FILE     = REPO_ROOT / "README.md"


class JournalGenerationError(Exception):
    pass


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise JournalGenerationError(
            "GEMINI_API_KEY is missing or empty. Set the GitHub Actions secret "
            "or export the environment variable before running this script."
        )
    return genai.Client(api_key=api_key)


def generate_content(client: genai.Client, *, prompt: str, temperature: float):
    try:
        return client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={"temperature": temperature},
        )
    except genai_errors.ClientError as exc:
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            raise JournalGenerationError(
                "Gemini quota was exceeded for this API key. Enable billing or use a key "
                "with available Gemini quota, then run the script again."
            ) from exc
        raise

# ── Curriculum ────────────────────────────────────────────────────────────────

TOPICS = [
    # Foundations
    "Perceptrons and the biological neuron analogy",
    "Activation functions: sigmoid, tanh, ReLU, Leaky ReLU, GELU",
    "Loss functions: MSE, Cross-Entropy, Huber loss",
    "Gradient Descent: batch, mini-batch, stochastic",
    "Backpropagation intuition and the chain rule",
    "Weight initialization strategies: Xavier, He, random",
    "Regularization: L1, L2, Dropout, BatchNorm",
    "Learning rate schedules: step decay, cosine annealing, warm restarts",
    # ANN
    "Feedforward Neural Networks (ANN) architecture",
    "Universal approximation theorem",
    "Vanishing and exploding gradients in deep ANNs",
    "Optimizers: SGD, Momentum, RMSProp, Adam, AdamW",
    "Hyperparameter tuning strategies",
    # CNN
    "Convolutional layers: filters, stride, padding",
    "Pooling layers: max pooling, average pooling",
    "CNN receptive field and spatial hierarchy",
    "Classic architectures: LeNet, AlexNet, VGG",
    "ResNets and skip connections",
    "Transfer learning and fine-tuning with CNNs",
    "Data augmentation for image tasks",
    "Object detection: YOLO, R-CNN family overview",
    # RNN
    "Recurrent Neural Networks (RNN) unrolled through time",
    "BPTT: Backpropagation Through Time",
    "Long Short-Term Memory (LSTM) gates explained",
    "Gated Recurrent Units (GRU)",
    "Sequence-to-sequence models and encoder-decoder",
    "Attention mechanisms: self-attention and cross-attention",
    "The Transformer architecture (Attention is All You Need)",
    "Positional encoding in Transformers",
    "BERT and masked language modeling",
    "GPT and autoregressive language modeling",
    # Advanced
    "Generative Adversarial Networks (GANs): generator vs discriminator",
    "Variational Autoencoders (VAEs)",
    "Diffusion models intuition",
    "Graph Neural Networks (GNNs) basics",
    "Reinforcement Learning foundations: MDP, rewards, policy",
    "Deep Q-Networks (DQN)",
    "Multi-task and meta-learning overview",
    "Neural Architecture Search (NAS)",
    "Quantization and model pruning for deployment",
    "Federated learning and privacy-preserving ML",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"day": 0, "completed_topics": [], "streak": 0, "last_date": ""}

def save_progress(p: dict):
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))

def pick_topic(progress: dict) -> str:
    done = set(progress.get("completed_topics", []))
    remaining = [t for t in TOPICS if t not in done]
    if not remaining:           # all topics done — cycle again
        remaining = TOPICS
        progress["completed_topics"] = []
    topic = remaining[0]        # ordered curriculum
    progress["completed_topics"].append(topic)
    return topic

# ── Gemini calls ─────────────────────────────────────────────────────────────

def generate_concept_notes(topic: str, day: int) -> dict:
    prompt = f"""You are an expert ML educator writing a daily deep learning journal entry.

Topic for Day {day}: **{topic}**

Return ONLY a valid JSON object (no markdown fences) with these exact keys:

{{
  "concept_explanation": "...",   // 200-250 words, clear intuitive explanation
  "key_points": ["...", "..."],   // exactly 5 crisp bullet points
  "architecture_ascii": "...",    // a simple ASCII diagram OR formula block
  "analogy": "...",               // one real-world analogy in 1-2 sentences
  "common_mistakes": ["...", "..."], // 3 common beginner mistakes
  "resources": [                  // 2 free resources
    {{"title": "...", "url": "...", "type": "paper|video|article"}}
  ],
  "quiz": [                       // 2 multiple-choice questions
    {{
      "q": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A"
    }}
  ]
}}"""

    client = get_client()
    resp = generate_content(client, prompt=prompt, temperature=0.7)
    raw = resp.text.strip()
    # strip accidental fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def generate_experiment_code(topic: str, day: int) -> str:
    prompt = f"""Write a self-contained Python mini-experiment for Day {day} of a deep learning journal.
Topic: {topic}

Requirements:

Return ONLY the raw Python code, no markdown fences."""

    client = get_client()
    resp = generate_content(client, prompt=prompt, temperature=0.4)
    code = resp.text.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
    return code

# ── Markdown builder ──────────────────────────────────────────────────────────

def build_markdown(day: int, topic: str, date_str: str, notes: dict, exp_filename: str) -> str:
    kp  = "\n".join(f"- {p}" for p in notes["key_points"])
    mis = "\n".join(f"- {m}" for m in notes["common_mistakes"])
    res = "\n".join(
        f"- [{r['title']}]({r['url']}) `{r['type']}`"
        for r in notes["resources"]
    )
    quiz_md = ""
    for i, q in enumerate(notes["quiz"], 1):
        opts = "\n".join(f"  {o}" for o in q["options"])
        quiz_md += f"\n**Q{i}.** {q['q']}\n{opts}\n\n<details><summary>Answer</summary>{q['answer']}</details>\n"

    return f"""# Day {day} — {topic}

> 📅 {date_str}

---

## 🧠 Concept Explanation

{notes['concept_explanation']}

---

## ✅ Key Points

{kp}

---

## 🏗️ Diagram / Formula

```
{notes['architecture_ascii']}
```

---

## 💡 Real-World Analogy

{notes['analogy']}

---

## ⚠️ Common Mistakes

{mis}

---

## 🧪 Mini Experiment

See [`experiments/{exp_filename}`](../experiments/{exp_filename}) — run it locally:

```bash
python journal/experiments/{exp_filename}
```

---

## 📚 Resources

{res}

---

## 🧩 Quick Quiz
{quiz_md}

---

*Generated by [AI Learning Journal](../../README.md) • Day {day} of ∞*
"""

# ── README updater ────────────────────────────────────────────────────────────

def update_readme(progress: dict, today_topic: str, today_date: str, day: int):
    total   = len(TOPICS)
    done    = len(progress["completed_topics"])
    pct     = int((done / total) * 100)
    bar_len = 30
    filled  = int(bar_len * pct / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)

    # build recent entries table (last 5)
    entries = sorted(ENTRIES_DIR.glob("day_*.md"), reverse=True)[:5]
    table_rows = ""
    for e in entries:
        parts = e.stem.split("_", 2)          # day_001_topic-slug
        d_num = parts[1] if len(parts) > 1 else "?"
        slug  = parts[2].replace("-", " ").title() if len(parts) > 2 else e.stem
        table_rows += f"| {d_num} | [{slug}](journal/entries/{e.name}) |\n"

    readme = f"""# 🧠 AI Learning Journal

> **Automated daily deep learning notes** — concepts, experiments & quizzes, generated with Gemini and pushed by GitHub Actions every day.

[![Update Journal](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml/badge.svg)](https://github.com/vineetm1204-m/ai-learning-journal/actions/workflows/daily_journal.yml)


## 📊 Progress

```
Day {day:>3} / ∞   Topics covered: {done}/{total}
[{bar}] {pct}%
🔥 Current streak: {progress['streak']} days
```


## 📅 Latest Entry

**Day {day} — {today_topic}**  
🗓️ {today_date}  
📖 [Read entry →](journal/entries/day_{day:03d}_{today_topic[:40].lower().replace(' ', '-').replace('/', '-')}.md)


## 📚 Recent Entries

| Day | Topic |
|-----|-------|
{table_rows}
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
pip install torch numpy google-genai
python journal/experiments/day_001_*.py
```


## ⚙️ How It Works

```
GitHub Actions (cron: daily 6 AM UTC)
        │
        ▼
generate_entry.py
        │
        ├── Gemini → concept notes (JSON)
        ├── Gemini → experiment code (.py)
        ├── Builds Markdown entry
        ├── Updates README.md
        └── git commit & push
```


*Built with ❤️ by Vineet Mittal*
"""
    README_FILE.write_text(readme)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today      = datetime.now(timezone.utc)
    date_str   = today.strftime("%B %d, %Y")
    date_slug  = today.strftime("%Y-%m-%d")

    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()

    # Update streak
    last = progress.get("last_date", "")
    if last == (today - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d"):
        progress["streak"] = progress.get("streak", 0) + 1
    elif last != date_slug:
        progress["streak"] = 1
    progress["last_date"] = date_slug
    progress["day"] = progress.get("day", 0) + 1
    day = progress["day"]

    topic = pick_topic(progress)
    print(f"📚 Day {day}: {topic}")

    print("  ⏳ Generating concept notes...")
    notes = generate_concept_notes(topic, day)

    print("  ⏳ Generating experiment code...")
    code  = generate_experiment_code(topic, day)

    # Slugify topic for filenames
    slug = topic[:50].lower()
    for ch in " /\\:*?\"<>|(),.":
        slug = slug.replace(ch, "-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")

    exp_filename   = f"day_{day:03d}_{slug}.py"
    entry_filename = f"day_{day:03d}_{slug}.md"

    # Write experiment
    exp_path = EXPERIMENTS_DIR / exp_filename
    exp_path.write_text(code)
    print(f"  ✅ Experiment → {exp_path}")

    # Write journal entry
    md = build_markdown(day, topic, date_str, notes, exp_filename)
    entry_path = ENTRIES_DIR / entry_filename
    entry_path.write_text(md)
    print(f"  ✅ Entry      → {entry_path}")

    # Update progress & README
    save_progress(progress)
    update_readme(progress, topic, date_str, day)
    print("  ✅ README.md updated")
    print(f"\n🎉 Day {day} complete! Streak: {progress['streak']} days 🔥")


if __name__ == "__main__":
    try:
        main()
    except JournalGenerationError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
