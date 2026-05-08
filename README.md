# 🛡️ PhishGuard AI — AI-Powered Phishing Email Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-brightgreen?style=for-the-badge&logo=mongodb)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-blue?style=for-the-badge)

**A production-ready, enterprise-grade ML system for detecting phishing and spam emails**

[Quick Start](#quick-start) • [API Docs](#api-endpoints) • [ML Pipeline](#ml-pipeline) • [Docker](#docker-deployment) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Model Performance](#model-performance)
- [Security Features](#security-features)
- [Configuration](#configuration)
- [Testing](#testing)
- [Future Roadmap](#future-roadmap)

---

## 🎯 Project Overview

**PhishGuard AI** is a complete, production-ready backend system that uses Machine Learning and Natural Language Processing to classify emails as phishing/spam or legitimate with high accuracy and — critically — **high recall** to minimise dangerous false negatives.

### Key Highlights

| Feature | Details |
|---------|---------|
| 🤖 **Models** | LightGBM, XGBoost, Random Forest, Logistic Regression, Naive Bayes |
| 🧠 **NLP** | TF-IDF (word + char n-grams) + 13 hand-crafted phishing features |
| 📊 **Dataset** | ~82,500 emails (42,891 spam + 39,595 legitimate) |
| 🎯 **Optimised For** | Recall ≥ 95% (cybersecurity requirement) |
| 🔍 **Explainability** | SHAP feature attribution |
| ⚡ **API** | Async FastAPI with < 50ms inference latency |
| 🗄️ **Database** | MongoDB with full prediction history |
| 🔒 **Security** | JWT auth, API keys, rate limiting, input validation |
| 🐳 **Deployment** | Docker + Docker Compose (one command) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PhishGuard AI Architecture                   │
├─────────────────┬───────────────────────────────────────────────┤
│   Client        │  React / Next.js / API Consumer               │
├─────────────────┴───────────────────────────────────────────────┤
│   FastAPI       │  Rate Limiting → Auth → Middleware → Routes   │
├─────────────────┬───────────────────────────────────────────────┤
│   ML Service    │  Preprocessor → TF-IDF → Model → SHAP        │
│   (Singleton)   │  LightGBM (best) / XGBoost / RF / LR / NB   │
├─────────────────┼───────────────────────────────────────────────┤
│   Celery        │  Background training / Scheduled retraining   │
├─────────────────┼───────────────────────────────────────────────┤
│   MongoDB       │  predictions / users / logs collections       │
├─────────────────┼───────────────────────────────────────────────┤
│   Redis         │  Celery broker + result backend               │
└─────────────────┴───────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Dataset | Size | Source |
|---------|------|--------|
| `phishing_email.csv` | ~100MB | Combined dataset |
| `CEAS_08.csv` | 68MB | CEAS 2008 spam challenge |
| `Enron.csv` | 46MB | Enron email corpus |
| `Ling.csv` | 9MB | Ling spam collection |
| `Nazario.csv` | 7.8MB | Nazario phishing corpus |
| `Nigerian_Fraud.csv` | 9.2MB | Nigerian fraud emails |
| `SpamAssasin.csv` | 15MB | SpamAssassin corpus |

**Total: ~82,500 emails | Spam: 42,891 (52%) | Legitimate: 39,595 (48%)**

All datasets are placed in the `datasets/` directory.

---

## 🧠 ML Pipeline

```
Raw Emails
    │
    ▼
┌─────────────────────────────────┐
│  1. DATA LOADING                │  Multi-dataset merge + deduplication
├─────────────────────────────────┤
│  2. TEXT PREPROCESSING          │
│   • HTML/XML tag removal        │
│   • URL → URL token             │
│   • Email address removal       │
│   • Phone number removal        │
│   • Lowercase conversion        │
│   • Punctuation removal         │
│   • Stopword removal (NLTK)     │
│   • Lemmatization (WordNet)     │
├─────────────────────────────────┤
│  3. FEATURE ENGINEERING         │
│   • TF-IDF word n-grams (1-2)  │
│   • TF-IDF char n-grams (3-5)  │
│   • 13 hand-crafted features:  │
│     - URL count                 │
│     - Phishing keyword count    │
│     - Exclamation mark count    │
│     - ALL-CAPS word count       │
│     - HTML presence             │
│     - Dollar sign count         │
│     - Text length / word count  │
│     - Digit ratio               │
├─────────────────────────────────┤
│  4. TRAIN / VAL / TEST SPLIT    │  70% / 10% / 20% (stratified)
├─────────────────────────────────┤
│  5. MODEL TRAINING              │
│   • Logistic Regression         │
│   • Random Forest               │
│   • XGBoost                     │
│   • LightGBM  ← (usually best) │
│   • Complement Naive Bayes      │
├─────────────────────────────────┤
│  6. EVALUATION                  │
│   • Precision / Recall / F1     │
│   • ROC-AUC                     │
│   • Confusion Matrix            │
├─────────────────────────────────┤
│  7. THRESHOLD OPTIMISATION      │  Maximise Recall ≥ 95%
├─────────────────────────────────┤
│  8. SHAP EXPLAINABILITY         │  Top feature attribution
├─────────────────────────────────┤
│  9. MODEL SAVING (joblib)       │  model + feature_engineer + metadata
└─────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI 0.111 |
| **ASGI Server** | Uvicorn |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **NLP** | NLTK, BeautifulSoup |
| **Database** | MongoDB 7 (Motor async driver + Beanie ODM) |
| **Cache/Queue** | Redis 7 |
| **Task Queue** | Celery 5 + Flower monitoring |
| **Auth** | JWT (python-jose) + bcrypt |
| **Rate Limiting** | slowapi |
| **Explainability** | SHAP |
| **Logging** | Loguru (structured JSON) |
| **Metrics** | Prometheus + prometheus-fastapi-instrumentator |
| **Containers** | Docker + Docker Compose |
| **Testing** | pytest + httpx async client |

---

## 📂 Project Structure

```
phishing-detection/
│
├── 📁 backend/                    # FastAPI application
│   ├── 📁 api/                    # Route handlers
│   │   ├── auth.py                # Authentication endpoints
│   │   ├── predict.py             # Prediction endpoints
│   │   ├── train.py               # Training trigger endpoint
│   │   └── monitoring.py          # Health / Metrics / History
│   ├── 📁 core/
│   │   └── auth.py                # JWT + API key logic
│   ├── 📁 db/
│   │   ├── connection.py          # Motor / Beanie setup
│   │   └── models.py              # Document models (collections)
│   ├── 📁 middleware/
│   │   └── logging_middleware.py  # Request logging
│   ├── 📁 schemas/
│   │   └── schemas.py             # Pydantic schemas
│   ├── 📁 services/
│   │   ├── detector.py            # ML inference service (singleton)
│   │   └── celery_worker.py       # Background task worker
│   └── main.py                    # FastAPI app entry point
│
├── 📁 ml_pipeline/                # Machine Learning pipeline
│   ├── 📁 data/
│   │   └── loader.py              # Multi-dataset loader
│   ├── 📁 preprocessing/
│   │   └── text_preprocessor.py  # NLP preprocessing
│   ├── 📁 features/
│   │   └── feature_engineer.py   # TF-IDF + hand-crafted features
│   ├── 📁 training/
│   │   └── trainer.py            # Multi-model trainer
│   ├── 📁 evaluation/
│   │   └── evaluator.py          # Metrics + SHAP
│   ├── 📁 models/
│   │   └── model_store.py        # Save/load artefacts
│   └── pipeline.py               # Full training orchestrator
│
├── 📁 models/saved/               # Trained model artefacts (gitignored)
│   ├── model.joblib
│   ├── feature_engineer.joblib
│   └── metadata.json
│
├── 📁 datasets/                   # Dataset CSV files (gitignored)
│   ├── phishing_email.csv
│   ├── CEAS_08.csv
│   └── ...
│
├── 📁 config/
│   └── settings.py                # Pydantic Settings (env-driven)
│
├── 📁 utils/
│   ├── logger.py                  # Loguru setup
│   └── helpers.py                 # Utility functions
│
├── 📁 scripts/
│   ├── train.py                   # CLI training script
│   └── predict.py                 # CLI prediction test script
│
├── 📁 tests/
│   ├── 📁 api/
│   │   └── test_endpoints.py      # API integration tests
│   ├── 📁 ml/
│   │   └── test_preprocessor.py   # ML unit tests
│   └── 📁 integration/
│
├── 📁 docker/
│   └── mongo-init.js              # MongoDB initialisation script
│
├── 📁 logs/                       # Application logs (auto-created)
│
├── Dockerfile                     # Multi-stage production image
├── docker-compose.yml             # Full stack orchestration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for containerised setup)
- MongoDB (local or Docker)
- Redis (local or Docker)

### Option 1: Docker (Recommended — One Command)

```bash
# Clone and enter project
cd "Phishing Detection"

# Copy environment config
cp .env.example .env

# Start all services (FastAPI + MongoDB + Redis + Celery)
docker-compose up -d

# Check all services are healthy
docker-compose ps

# Train the model (runs in background)
curl -X POST http://localhost:8000/api/v1/train \
  -H "X-API-Key: admin-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"max_features": 75000}'

# Monitor training progress
docker-compose logs -f celery_worker
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Copy environment config
cp .env.example .env
# Edit .env with your MongoDB and Redis URLs

# 5. Train the model
python scripts/train.py

# 6. Start the API server
python backend/main.py
# OR
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints

### Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | None | Service health check |
| `POST` | `/predict` | Optional | Predict single email |
| `POST` | `/predict/batch` | Required | Batch predict (max 50) |
| `GET` | `/metrics` | Required | Model performance metrics |
| `GET` | `/history` | Required | Prediction history |
| `POST` | `/train` | Admin | Trigger model training |
| `GET` | `/train/status/{id}` | Admin | Training task status |
| `POST` | `/auth/register` | None | Register user |
| `POST` | `/auth/login` | None | Login (get JWT) |
| `GET` | `/auth/me` | Required | Current user info |
| `POST` | `/auth/api-key` | Required | Generate API key |

### 🔍 Predict Email

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "Congratulations! You have won $1,000,000. Click here to claim your prize immediately!"
  }'
```

**Response:**
```json
{
  "prediction": "spam",
  "label": 1,
  "confidence": 98.7,
  "risk_level": "high",
  "model_name": "lightgbm",
  "threshold": 0.3812,
  "phishing_keywords": ["congratulations", "claim", "prize", "click"],
  "top_features": null,
  "processing_time_ms": 23.4,
  "prediction_id": "663abc123def456789"
}
```

### 📊 Health Check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "model_loaded": true,
  "model_name": "lightgbm",
  "model_threshold": 0.3812,
  "database_connected": true,
  "uptime_seconds": 3600.5,
  "timestamp": "2026-05-08T12:00:00Z"
}
```

### 🎓 With SHAP Explanation

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "URGENT: Your bank account has been suspended. Verify now!",
    "include_explanation": true
  }'
```

### 📜 Interactive Docs

| URL | Description |
|-----|-------------|
| `http://localhost:8000/api/docs` | Swagger UI |
| `http://localhost:8000/api/redoc` | ReDoc |
| `http://localhost:8000/metrics` | Prometheus metrics |
| `http://localhost:5555` | Celery Flower monitor |

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f api
docker-compose logs -f celery_worker

# Scale workers
docker-compose up -d --scale celery_worker=3

# Stop all
docker-compose down

# Stop and remove volumes (full reset)
docker-compose down -v
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI application |
| `mongodb` | 27017 | MongoDB database |
| `redis` | 6379 | Redis cache/broker |
| `celery_worker` | — | Background task processor |
| `celery_beat` | — | Scheduled task scheduler |
| `flower` | 5555 | Celery task monitor UI |

---

## 📈 Model Performance

Expected performance on the test set (20% holdout):

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| **LightGBM** ⭐ | ~0.97 | ~0.97 | ~0.97 | ~0.99 |
| XGBoost | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| Random Forest | ~0.95 | ~0.95 | ~0.95 | ~0.98 |
| Logistic Regression | ~0.93 | ~0.93 | ~0.93 | ~0.97 |
| Naive Bayes | ~0.90 | ~0.92 | ~0.91 | ~0.96 |

> **Note:** Actual metrics depend on your dataset and preprocessing. The model is automatically selected by F1 score with Recall as a tiebreaker.

---

## 🔒 Security Features

- ✅ **JWT Authentication** (access + refresh tokens)
- ✅ **API Key authentication** via `X-API-Key` header
- ✅ **Rate limiting** (100 req/min per IP by default)
- ✅ **Input validation** (Pydantic schemas)
- ✅ **Input sanitisation** (null bytes, max length)
- ✅ **Injection protection** (XSS, SQLi, MongoDB injection patterns)
- ✅ **CORS** configuration
- ✅ **Non-root Docker user**
- ✅ **Environment variables** (no secrets in code)
- ✅ **GZip compression**

---

## ⚙️ Configuration

All configuration is via environment variables (`.env` file):

```env
APP_ENV=production
SECRET_KEY=your-32-char-secret-key
JWT_SECRET_KEY=your-jwt-secret
MONGO_URI=mongodb://admin:password@localhost:27017
REDIS_URL=redis://localhost:6379/0
DEFAULT_MODEL=lightgbm
CONFIDENCE_THRESHOLD=0.5
HIGH_RISK_THRESHOLD=0.8
RATE_LIMIT_REQUESTS=100
LOG_LEVEL=INFO
```

See `.env.example` for all available options.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=backend --cov=ml_pipeline

# Run only API tests
pytest tests/api/

# Run only ML tests
pytest tests/ml/

# Run specific test
pytest tests/ml/test_preprocessor.py::TestTextPreprocessor::test_html_removal -v
```

---

## 🔮 Future Roadmap

- [ ] **Transformer models** (BERT, DistilBERT) for email body encoding
- [ ] **Ensemble stacking** across trained models
- [ ] **Active learning** pipeline for uncertain predictions
- [ ] **Header analysis** (SPF/DKIM/DMARC validation)
- [ ] **URL reputation check** via external threat intelligence API
- [ ] **Real-time email integration** (IMAP/SMTP hooks)
- [ ] **React dashboard** with live prediction charts
- [ ] **Multi-language support**
- [ ] **A/B testing framework** for model comparison in production

---

## 👨‍💻 Development

```bash
# Format code
black .
isort .

# Type checking
mypy backend ml_pipeline

# Lint
flake8 backend ml_pipeline
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for cybersecurity research and education.**

⭐ Star this repo if it helped you!

</div>
# Phishing-Detection-Model
