# 🚀 Setup Guide

Follow these steps to get your AI Learning Journal running in ~5 minutes.

---

## Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `ai-learning-journal` (or anything you like)
3. Set it to **Public** (so your progress is visible on your profile)
4. **Do NOT** initialize with README — you'll push this folder

---

## Step 2 — Push this project

```bash
cd ai-learning-journal
git init
git add .
git commit -m "🚀 Initial setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-learning-journal.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 3 — Add your Gemini API Key as a secret

1. Open your repo on GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Name: `GEMINI_API_KEY`
5. Value: your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
6. Click **Add secret**

---

## Step 4 — Update your username in the README template

In `scripts/generate_entry.py`, find the two occurrences of:

```
YOUR_USERNAME
```

Replace with your actual GitHub username (used for badge URLs).

---

## Step 5 — Run your first entry manually

To generate Day 1 immediately (without waiting for the cron schedule):

**Option A — GitHub UI (easiest)**
1. Go to your repo → **Actions** tab
2. Click **🧠 Daily AI Learning Journal**
3. Click **Run workflow** → **Run workflow**

**Option B — Locally**
```bash
pip install google-genai
export GEMINI_API_KEY=your_key
python scripts/generate_entry.py
git add . && git commit -m "Day 1" && git push
```

---

## Step 6 — Verify the schedule

The workflow runs every day at **6:00 AM UTC** automatically.

You can change the time in `.github/workflows/daily_journal.yml`:

```yaml
- cron: "0 6 * * *"   # 6 AM UTC daily
```

Use [crontab.guru](https://crontab.guru) to pick your preferred time.

---

## 📁 What gets generated each day

```
journal/
├── entries/
│   └── day_001_activation-functions.md    ← full notes + quiz
├── experiments/
│   └── day_001_activation-functions.py    ← runnable Python code
└── progress.json                          ← streak + completed topics
README.md                                  ← auto-updated dashboard
```

---

## 💰 Cost estimate

Each daily run makes 2 Gemini calls.

| | Cost |
|--|--|
| Per day | ~$0.001 |
| Per month | ~$0.03 |

Essentially free. 🎉

---

## ❓ FAQ

**The workflow failed — what do I check?**
- Go to Actions tab → click the failed run → expand steps for error details
- Most common cause: `GEMINI_API_KEY` secret not set correctly

**Can I change the topic curriculum?**
Edit the `TOPICS` list in `scripts/generate_entry.py` — add, remove, or reorder freely.

**What if I miss a day?**
The streak resets but your entries continue. Manually trigger the workflow anytime.
