# 🧠 SentiPulse: Hybrid Social Media Sentiment Analytics & ML Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![NLTK](https://img.shields.io/badge/NLTK-VADER-154726?style=for-the-badge&logo=nltk&logoColor=white)](https://nltk.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**SentiPulse** is a production-ready, full-stack sentiment analytics platform that pairs **NLTK VADER lexical valence scoring** with a **Naive Bayes Machine Learning classifier** trained on 74,000+ social media records across 32 entity categories.

Built to address the challenge of evaluating domain-specific social media text, SentiPulse combines rule-based lexical analysis with statistical machine learning to produce consensus sentiment diagnostics with sub-5ms response latency.

---

## 📸 Key Features

- **⚡ Real-Time Live Diagnostics**: Input social media text, tweets, or product reviews to receive instant hybrid sentiment classification (`Positive`, `Negative`, `Neutral`), VADER compound scores, Naive Bayes probability confidence, and sub-5ms latency metrics.
- **🔍 Signal Extraction**: Automatically highlights domain unigram feature signals (e.g. `awesome`, `freezing`, `lagging`) that influenced the Naive Bayes prediction model.
- **📊 Interactive Entity Analytics**: Explore real-time sentiment distribution metrics across 32 major entities (e.g. *Borderlands*, *Overwatch*, *Microsoft*, *Amazon*) using responsive Chart.js doughnut and bar visualizers.
- **📈 Dynamic Agreement Tracking**: Computes real-time agreement metrics comparing lexical rule predictions against trained ML predictions across sampling sets.
- **📑 Multi-Line Batch Evaluator**: Submit bulk text lists (up to 50 items per request) for parallel sentiment analysis, complete with real-time progress, summary breakdown charts, and **CSV / JSON export**.
- **🔌 RESTful API Suite**: Exposes lightweight JSON endpoints (`/api/health`, `/api/analyze`, `/api/stats`, `/api/entity/<name>`, `/api/batch`).
- **🎨 Glassmorphic Dark-Mode UI**: Built with responsive HTML5, Vanilla CSS3, Google Fonts (*Outfit* & *Inter*), micro-animations, and full mobile drawer navigation.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend & REST API** | Python 3.9+, Flask 3.0 | Lightweight Web Server & Microservices API |
| **Lexical Engine** | NLTK VADER | Rule-based valence scoring (boosters, negations, punctuation) |
| **Machine Learning** | NLTK Naive Bayes Classifier | Statistical ML model trained on top unigram features |
| **Data Processing** | Python `csv`, `Counter`, `re` | Memory-efficient CSV parsing and regex token cleaning |
| **Frontend UI** | HTML5, Vanilla CSS3, ES6 JS | Responsive Glassmorphic Dashboard UI |
| **Data Visualization** | Chart.js 4.x | Interactive doughnut & bar charts |
| **Production WSGI** | Gunicorn | Production HTTP server for Linux / Cloud deployments |

---

## 🏗️ System Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   Client Dashboard (Browser)                     │
│         HTML5 / CSS3 / ES6 JS / Chart.js Data Visuals            │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ HTTP / REST API (JSON)
┌────────────────────────────────▼─────────────────────────────────┐
│                    Flask Controller Layer (app.py)               │
└─────────────────┬───────────────────────────────┬────────────────┘
                  │                               │
┌─────────────────▼─────────────┐   ┌─────────────▼────────────────┐
│   VADER Lexical Engine        │   │   Naive Bayes ML Model       │
│ - Lexicon Valence Dictionary  │   │ - Text Sanitization          │
│ - Negation & Booster Rules    │   │ - Stopword & Noise Filtering │
│ - Compound Scale [-1.0, +1.0] │   │ - Top 400 Unigram Features   │
└─────────────────┬─────────────┘   └─────────────┬────────────────┘
                  │                               │
┌─────────────────▼───────────────────────────────▼────────────────┐
│              Hybrid Consensus Engine & Latency Metric            │
│  Evaluates rule confidence vs ML probabilities -> Output JSON    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
sentipulse/
├── api/
│   └── index.py                # WSGI entry point for Vercel Serverless
├── nltk_data/
│   └── sentiment/
│       └── vader_lexicon.zip   # Bundled offline VADER lexicon dataset
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphic CSS styling & responsive grid
│   └── js/
│       └── main.js             # Async API client, Chart.js rendering & CSV/JSON export
├── templates/
│   └── index.html              # Single-page dashboard application HTML template
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules
├── app.py                      # Flask backend routes, health check & API endpoints
├── dataset_loader.py           # Memory-efficient CSV parser & dataset metrics calculator
├── Procfile                    # Deployment manifest for Heroku / Render
├── README.md                   # Project documentation
├── render.yaml                 # Infrastructure configuration for Render hosting
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Target Python runtime version specification
├── sentiment_engine.py         # Hybrid sentiment engine combining VADER & Naive Bayes
├── twitter_training.csv        # 74,000+ social media sentiment records
└── vercel.json                 # Vercel serverless deployment routing config
```

---

## 🧠 NLP & Machine Learning Pipeline

SentiPulse implements a **Hybrid NLP Pipeline** that overcomes the limitations of using a rule-based engine or a statistical model in isolation:

1. **Preprocessing & Tokenization**:
   - **VADER Path**: Keeps raw text intact (preserving capitalization, exclamation marks, and emoticons which heavily influence VADER valence).
   - **Naive Bayes ML Path**: Cleans text using regex (stripping URLs, `@` mentions, special symbols, numbers) and filters out 120+ standard English stopwords to isolate sentiment-bearing unigrams.

2. **Lexical Valence Scoring (VADER)**:
   - Calculates `pos`, `neg`, and `neu` proportions alongside a normalized `compound` score from `-1.0` (most negative) to `+1.0` (most positive).

3. **Statistical Machine Learning (Naive Bayes)**:
   - Trains a `NaiveBayesClassifier` on a balanced subset of labeled social media records.
   - Extracts the top 400 most informative word features (e.g. `contains(awesome)=True`).
   - Computes posterior class probabilities and returns prediction label + confidence probability.

4. **Consensus Aggregation**:
   - If VADER and Naive Bayes agree, consensus label is assigned with high confidence.
   - If VADER compound score shows extreme valence (`|compound| >= 0.4`), VADER's rule engine takes precedence.
   - For subtle or ambiguous text, the domain-trained Naive Bayes classifier breaks tie decisions.

---

## 🔌 REST API Reference

### 1. Health Check
`GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "records_count": 10000,
  "ml_trained": true,
  "features_count": 400
}
```

---

### 2. Single Text Analysis
`POST /api/analyze`

**Request:**
```json
{
  "text": "SentiPulse is an awesome platform! The results are fantastic."
}
```

**Response:**
```json
{
  "consensus_label": "Positive",
  "text": "SentiPulse is an awesome platform! The results are fantastic.",
  "vader": {
    "compound": 0.8398,
    "label": "Positive",
    "neg": 0.0,
    "neu": 0.467,
    "pos": 0.533
  },
  "ml": {
    "confidence": 0.543,
    "label": "Positive",
    "matched_features": ["awesome", "fantastic"]
  },
  "processing_time_ms": 1.35
}
```

---

### 3. Dataset Overview Stats
`GET /api/stats`

**Response:**
```json
{
  "total_records": 10000,
  "unique_entities": 32,
  "sentiment_distribution": {
    "Positive": 3239,
    "Neutral": 2596,
    "Negative": 2303,
    "Irrelevant": 1862
  },
  "top_entities": [["Borderlands", 2279], ["Overwatch", 2312]],
  "vader_ml_agreement_rate": 88.5
}
```

---

### 4. Entity Sentiment Breakdown
`GET /api/entity/<name>`

**Response:**
```json
{
  "entity": "Borderlands",
  "total_tweets": 2279,
  "distribution": { "Positive": 1020, "Negative": 540, "Neutral": 719 },
  "samples": [
    { "text": "Borderlands is amazing!", "sentiment": "Positive" }
  ]
}
```

---

### 5. Multi-Line Batch Analysis
`POST /api/batch`

**Request:**
```json
{
  "texts": [
    "Great product!",
    "Terrible service and server crashes."
  ]
}
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` to customize settings locally:

```env
PORT=5000
FLASK_ENV=development
FLASK_DEBUG=False
MAX_DATASET_RECORDS=10000
ML_SAMPLE_SIZE=1500
ML_TOP_FEATURES=400
```

---

## 🚀 Installation & Running Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pardhu643/sentipulse.git
   cd sentipulse
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Development Server**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your web browser.

---

## 🌐 Production Deployment

### Option 1: Deploy on Render
1. Connect your GitHub repository `Pardhu643/sentipulse` on [Render](https://render.com).
2. Set Environment: `Python 3`
3. Set Build Command: `pip install -r requirements.txt`
4. Set Start Command: `gunicorn app:app`

### Option 2: Deploy on Vercel
1. Install Vercel CLI or import repo on [Vercel Dashboard](https://vercel.com).
2. Vercel will automatically use `vercel.json` and build serverless functions via `@vercel/python`.

---

## 🔮 Future Improvements

- [ ] **Transformer Fine-Tuning**: Integrate a lightweight DistilBERT endpoint for deep contextual embeddings alongside VADER.
- [ ] **Real-Time Streaming**: Add WebSocket endpoint for live Twitter / Reddit post ingestion streams.
- [ ] **Aspect-Based Sentiment Analysis (ABSA)**: Extract sentiment target words for specific feature aspects (e.g. *graphics*, *gameplay*, *pricing*).

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
