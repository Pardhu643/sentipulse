# 🧠 SentiPulse: Hybrid Social Media Sentiment Analytics & ML Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![NLTK](https://img.shields.io/badge/NLTK-VADER-154726?style=for-the-badge&logo=nltk&logoColor=white)](https://nltk.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

SentiPulse is a full-stack, production-grade sentiment analytics platform that combines **NLTK VADER lexical valence scoring** with a **Naive Bayes Machine Learning classifier** trained on 74,000+ social media records across 32 entity categories.

---

## ✨ Features

- **⚡ Real-Time Live Diagnostics**: Input any tweet or text to receive instant VADER compound scores (`pos`, `neg`, `neu`, `compound`), confidence scores, and Naive Bayes ML classification labels.
- **📊 Interactive Entity Analytics**: Explore sentiment distribution metrics for 30+ major entities (e.g. *Borderlands*, *Overwatch*, *Microsoft*) with responsive Chart.js donut and bar charts.
- **📑 Multi-Line Batch Evaluation**: Submit bulk text lists for simultaneous batch sentiment extraction.
- **🔌 RESTful API Suite**: Exposes endpoints (`/api/analyze`, `/api/stats`, `/api/entity/<name>`, `/api/batch`) delivering JSON diagnostics with sub-100ms response latency.
- **🎨 Glassmorphic Dark-Mode UI**: Built with HTML5, Vanilla CSS3, Google Fonts (*Outfit* & *Inter*), and smooth micro-animations.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Client Interface (Browser)                  │
│       Glassmorphic Dashboard (HTML5 / CSS3 / Chart.js)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / REST API
┌──────────────────────────────▼──────────────────────────────┐
│                    Flask Web Server (app.py)                │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐┌──────────────▼──────────────┐
│   VADER Lexical Analyzer    ││   Naive Bayes ML Classifier  │
│ - Lexicon Valence           ││ - 1,500 Unigram Features    │
│ - Negation & Idiom Engine   ││ - Trained on 74k Tweets     │
└──────────────┬──────────────┘└──────────────┬──────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼──────────────┐
│             Consensus Aggregator & JSON Output              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Local Installation & Running

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pardhu643/sentipulse.git
   cd sentipulse
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Application**:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

---

## 🚀 Cloud Deployment (Render / Hugging Face)

This project contains a `Procfile`, `runtime.txt`, and `requirements.txt` pre-configured for 1-click hosting:

### Deploy to Render
1. Connect your GitHub repository `Pardhu643/sentipulse` on [Render.com](https://render.com).
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `gunicorn app:app`

---

## 📄 License
This project is licensed under the MIT License.
