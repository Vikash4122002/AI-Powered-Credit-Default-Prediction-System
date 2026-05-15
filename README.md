<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=6C63FF&center=true&vCenter=true&width=800&lines=AI+Credit+Default+Prediction+System;XGBoost+%7C+LightGBM+%7C+Stacking+Ensemble;FastAPI+%7C+Docker+%7C+AWS+EC2+%7C+Streamlit;End-to-End+MLOps+Deployment" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com/)
[![AWS](https://img.shields.io/badge/AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-0078D4?style=for-the-badge&logo=python&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-2ECC71?style=for-the-badge&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)

<br/>

> **30,000 credit card clients · Stacking Ensemble ML · FastAPI REST API · Dockerized · Deployed on AWS EC2 · 76.2% ROC-AUC**

<br/>

[![Live API](https://img.shields.io/badge/🚀_Live_API-3.109.32.46:8000-brightgreen?style=for-the-badge)](http://3.109.32.46:8000)
[![Swagger Docs](https://img.shields.io/badge/📖_Swagger_Docs-Available-blue?style=for-the-badge)](http://3.109.32.46:8000/docs)
[![Health Check](https://img.shields.io/badge/💚_Health_Check-Online-success?style=for-the-badge)](http://3.109.32.46:8000/health)
[![Docker Hub](https://img.shields.io/badge/🐳_Docker_Hub-vikash4122002-2496ED?style=for-the-badge)](https://hub.docker.com/r/vikash4122002/credit-default-api)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo Links](#-live-demo-links)
- [Key Results at a Glance](#-key-results-at-a-glance)
- [System Architecture](#-system-architecture)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [API Endpoints](#-api-endpoints)
- [Docker Deployment](#-docker-deployment)
- [AWS EC2 Deployment](#-aws-ec2-deployment)
- [Streamlit Frontend](#-streamlit-frontend)
- [Business Impact](#-business-impact)
- [Tech Stack & Skills Used](#-tech-stack--skills-used)
- [Project Structure](#-project-structure)
- [Recruiter Demo Guide](#-recruiter-demo-guide)
- [Installation & Setup](#-installation--setup)
- [Author](#-author)

---

## 🎯 Project Overview

This is a **production-grade, end-to-end MLOps project** that predicts credit card default risk using an advanced Stacking Ensemble combining XGBoost, LightGBM, and Random Forest. The trained model is served via a **FastAPI REST API**, containerized with **Docker**, deployed on **AWS EC2**, and accessible through an interactive **Streamlit frontend**.

This project mirrors real-world financial risk workflows used at banks and fintech companies — going from raw data to a live, scalable prediction API.

| Dimension | Detail |
|---|---|
| **Dataset** | UCI Credit Card Default Dataset (30,000 clients) |
| **Target** | Predict whether a client will default next month |
| **ML Approach** | Stacking Ensemble (RF + XGBoost + LightGBM + Meta XGBoost) |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **Tuning** | Optuna Bayesian Hyperparameter Optimization |
| **Backend** | FastAPI + Uvicorn |
| **Deployment** | Docker + AWS EC2 |
| **Frontend** | Streamlit + Plotly |
| **Business Goal** | Minimize false negatives — catch every likely defaulter |

---

## 🔗 Live Demo Links

> **All links are live and publicly accessible:**

| Resource | URL |
|---|---|
| 🚀 **Live API Base** | [http://3.109.32.46:8000](http://3.109.32.46:8000) |
| 📖 **Swagger UI (Interactive API Docs)** | [http://3.109.32.46:8000/docs](http://3.109.32.46:8000/docs) |
| 💚 **Health Monitoring Endpoint** | [http://3.109.32.46:8000/health](http://3.109.32.46:8000/health) |
| 🐳 **Docker Hub Image** | `docker pull vikash4122002/credit-default-api:latest` |
| 💻 **GitHub Repository** | [github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System](https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System) |

---

## 🏆 Key Results at a Glance

<div align="center">

| Metric | Value | Notes |
|---|---|---|
| 🧠 **Best Model** | Stacking Ensemble | RF + XGBoost + LightGBM + Meta XGBoost |
| 📈 **ROC-AUC** | **76.2%** | Key metric for imbalanced classification |
| 🎯 **Recall** | **57.1%** | Most critical — catching actual defaulters |
| ⚡ **Precision** | **46.8%** | Minimizing false positives |
| 📊 **F1 Score** | **51.4%** | Harmonic mean of precision & recall |
| 🔢 **Accuracy** | **76.2%** | At threshold = 0.30 |
| ⚙️ **Threshold** | **0.30** | Custom tuned to maximize recall |
| 📦 **Deployment** | **AWS EC2 + Docker** | Live, production-grade API |
| 🔄 **SMOTE** | Applied | Balanced class imbalance on training set |
| 🔍 **Tuning** | **Optuna (90 trials)** | 30 trials per base model |

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│         Single Prediction | Batch | Analytics           │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP REST API
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND                        │
│    /health  |  /api/v1/predict  |  /api/v1/predict_batch│
│              Swagger Docs at /docs                      │
└───────────────────────┬─────────────────────────────────┘
                        │ Loads
                        ▼
┌─────────────────────────────────────────────────────────┐
│              STACKING ENSEMBLE MODEL                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Random Forest│  │   XGBoost    │  │  LightGBM    │  │
│  │  (Optuna)    │  │  (Optuna)    │  │  (Optuna)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └─────────────────┼──────────────────┘          │
│                           ▼                             │
│              ┌────────────────────────┐                 │
│              │  Meta Learner XGBoost  │                 │
│              │       (Optuna)         │                 │
│              └────────────────────────┘                 │
└───────────────────────┬─────────────────────────────────┘
                        │ Containerized in
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  DOCKER CONTAINER                        │
│            vikash4122002/credit-default-api              │
└───────────────────────┬─────────────────────────────────┘
                        │ Hosted on
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    AWS EC2 INSTANCE                      │
│         Public IP: 3.109.32.46  |  Port: 8000           │
│              Globally Accessible API                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Pipeline

### Step 1 — Data Preprocessing

- **Dataset:** UCI Credit Card Default (30,000 clients, 23 features)
- **Cleaning:** Fixed invalid `EDUCATION` (0,5,6 → 4) and `MARRIAGE` (0 → 3) categories
- **Missing Values:** Imputed with column medians
- **Duplicates:** Removed duplicate rows
- **Train/Test Split:** 80/20 stratified split (to preserve class ratio)
- **Feature Scaling:** `StandardScaler` fitted only on training data (no data leakage)
- **SMOTE:** Applied exclusively on training set to balance 22% default rate

### Step 2 — Base Model Training with Optuna

| Model | Optuna Trials | Tuned Parameters |
|---|---|---|
| 🌲 Random Forest | 30 | n_estimators, max_depth, min_samples_split, max_features, bootstrap |
| ⚡ XGBoost | 30 | learning_rate, max_depth, subsample, colsample_bytree, reg_alpha/lambda |
| 🔦 LightGBM | 30 | num_leaves, learning_rate, subsample, colsample_bytree, reg_alpha/lambda |

### Step 3 — Stacking Ensemble

```
Base Models (Level 0): Random Forest + XGBoost + LightGBM
        ↓ (predict_proba outputs become new features)
Meta Learner (Level 1): XGBoost tuned with Optuna (20 trials)
        ↓
Final Prediction with custom threshold (0.30)
```

- **Cross-validation:** 5-Fold StratifiedKFold throughout
- **Stack method:** `predict_proba` (probability outputs passed to meta learner)
- **Passthrough:** False (only base model outputs fed to meta learner)

### Step 4 — Threshold Optimization

Custom threshold search from 0.05 to 0.65 with weighted scoring:

```python
score = 0.7 * recall + 0.3 * precision
```

> **Why 0.30 threshold?** In credit default prediction, false negatives (missing a defaulter) are far more costly than false positives. A lower threshold maximizes recall — catching more actual defaulters — at an acceptable precision trade-off.

---

## 📊 Model Performance

### Comparison at Threshold = 0.30

| Model | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|
| 🥇 **Stacking Ensemble** | **57.1%** | **46.8%** | **51.4%** | **76.2%** |
| ⚡ XGBoost | 55.8% | 47.2% | 51.1% | 75.8% |
| 🔦 LightGBM | 54.9% | 47.5% | 51.0% | 75.5% |
| 🌲 Random Forest | 52.3% | 48.1% | 50.1% | 74.9% |

### Why Recall Is the Priority Metric

```
Business Context: Bank Credit Risk Department

FALSE NEGATIVE (miss a defaulter):  → Bank loses full loan amount
FALSE POSITIVE (flag a non-defaulter): → Customer inconvenienced, bank loses sale

Cost of False Negative >> Cost of False Positive

Therefore: MAXIMIZE RECALL — catch as many defaulters as possible
```

---

## 🔌 API Endpoints

### Base URL
```
http://3.109.32.46:8000
```

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Interactive Swagger Docs
```http
GET /docs
```
> Full interactive API documentation — try predictions directly in the browser.

### 3. Single Prediction
```http
POST /api/v1/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [
    50000, 1, 2, 1, 35,
    0, 0, 0, 0, 0, 0,
    10000, 0, 0, 0, 0, 0,
    5000, 0, 0, 0, 0, 0
  ],
  "threshold": 0.3
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.187,
  "risk_level": "LOW",
  "threshold_used": 0.3
}
```

### 4. Batch Prediction
```http
POST /api/v1/predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    [50000, 1, 2, 1, 35, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 5000, 0, 0, 0, 0, 0],
    [20000, 2, 3, 2, 28, 2, 2, 0, 0, 0, 0, 15000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ],
  "threshold": 0.3
}
```

**Feature Order (23 features):**

| # | Feature | Description |
|---|---|---|
| 1 | LIMIT_BAL | Credit limit (NT dollars) |
| 2 | SEX | 1=Male, 2=Female |
| 3 | EDUCATION | 1=Graduate, 2=University, 3=High School, 4=Others |
| 4 | MARRIAGE | 1=Married, 2=Single, 3=Others |
| 5 | AGE | Age in years |
| 6–11 | PAY_0 to PAY_6 | Repayment status (-2=no consumption, -1=paid duly, 1–9=months delayed) |
| 12–17 | BILL_AMT1–6 | Bill statement amounts (Apr–Sep) |
| 18–23 | PAY_AMT1–6 | Previous payment amounts (Apr–Sep) |

---

## 🐳 Docker Deployment

### Pull & Run from Docker Hub

```bash
# Pull the image
docker pull vikash4122002/credit-default-api:latest

# Run the container
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest

# Verify it's running
docker ps

# Test the API
curl http://localhost:8000/health
```

### Build Locally

```bash
# Clone the repository
git clone https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System.git
cd AI-Powered-Credit-Default-Prediction-System/deployment_api

# Build Docker image
docker build -t credit-default-api .

# Run container
docker run -d -p 8000:8000 credit-default-api

# View logs
docker logs $(docker ps -q --filter "ancestor=credit-default-api")
```

### Dockerfile Overview

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ☁️ AWS EC2 Deployment

### Instance Configuration

| Setting | Value |
|---|---|
| **Instance Type** | t2.micro (Free Tier) |
| **OS** | Ubuntu 22.04 LTS |
| **Public IP** | 3.109.32.46 |
| **Port** | 8000 (open in Security Group) |
| **Region** | ap-south-1 (Mumbai) |

### Deployment Steps on EC2

```bash
# 1. SSH into EC2
ssh -i "your-key.pem" ubuntu@3.109.32.46

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# 3. Pull and run the container
sudo docker pull vikash4122002/credit-default-api:latest
sudo docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest

# 4. Verify deployment
sudo docker ps
curl http://localhost:8000/health

# 5. Keep running after logout
sudo docker run -d --restart=always -p 8000:8000 vikash4122002/credit-default-api:latest
```

---

## 🖥️ Streamlit Frontend

### Features

- **Single Prediction Tab** — Enter customer details, adjust risk threshold, get instant prediction with gauge chart
- **Batch Prediction Tab** — Upload CSV, run bulk predictions, download results
- **Analytics Tab** — View model performance metrics and bar charts
- **API Docs Tab** — Integrated API documentation with sample requests

### Run Locally

```bash
cd streamlit_frontend
pip install -r requirements.txt
streamlit run app.py
```

### Frontend Highlights

```
✅ Real-time prediction with probability gauge
✅ Risk level: LOW / MEDIUM / HIGH
✅ Threshold slider for risk adjustment
✅ CSV batch upload & download results
✅ API status indicator (Online/Offline)
✅ Interactive Plotly visualizations
✅ Mobile-responsive layout
```

---

## 💡 Business Impact

### Problem Statement

Banks lose billions annually from credit defaults. Traditional rule-based systems miss subtle default patterns. This ML system:

1. **Identifies high-risk customers before default occurs** — enabling proactive intervention
2. **Reduces false negatives** — catching more actual defaulters (57.1% recall)
3. **Scales to batch processing** — evaluate thousands of applications simultaneously
4. **Provides explainable risk levels** — actionable output for credit officers

### Financial Impact Estimation

```
Dataset: 30,000 clients | Default rate: 22.1% | Avg default loss: ~$10,000

Without ML system:
  → All defaults missed = 6,630 defaults × $10,000 = $66.3M exposure

With Stacking Ensemble (57.1% recall):
  → 3,786 defaults caught = $37.9M in losses prevented

Net Business Value of Model: ~$37.9M on 30K customer portfolio
```

---

## 🛠️ Tech Stack & Skills Used

| Category | Tool / Technology | Usage |
|---|---|---|
| **ML Framework** | Scikit-learn | Pipeline, stacking, metrics |
| **Boosting** | XGBoost | Base model + meta learner |
| **Boosting** | LightGBM | Base model |
| **Ensemble** | StackingClassifier | Combines base models |
| **Tuning** | Optuna (TPE Sampler) | Bayesian hyperparameter optimization |
| **Imbalance** | SMOTE (imbalanced-learn) | Synthetic oversampling |
| **Scaling** | StandardScaler | Feature normalization |
| **Backend** | FastAPI + Uvicorn | REST API with async support |
| **Validation** | Pydantic | Request/response schema |
| **Frontend** | Streamlit | Interactive web dashboard |
| **Visualization** | Plotly | Gauge charts, histograms |
| **Containerization** | Docker | Full app containerization |
| **Registry** | Docker Hub | Public image hosting |
| **Cloud** | AWS EC2 | Production deployment |
| **Data** | Pandas, NumPy | Data processing |
| **Visualization** | Matplotlib, Seaborn | EDA and evaluation plots |
| **Serialization** | Joblib | Model persistence |
| **Version Control** | Git + GitHub | Source control |

---

## 📁 Project Structure

```
📦 AI-Powered-Credit-Default-Prediction-System
│
├── 📂 deployment_api/
│   ├── 🐍 app.py                          ← FastAPI main application
│   ├── 🐳 Dockerfile                      ← Docker build file
│   ├── 📋 requirements.txt
│   ├── 📂 routes/
│   │   └── predict.py                     ← API route handlers
│   ├── 📂 services/
│   │   └── model_service.py               ← Model loading & inference
│   ├── 📂 schema/
│   │   └── request_schema.py              ← Pydantic request/response models
│   ├── 📂 utils/
│   │   └── preprocessing.py               ← Input preprocessing helpers
│   └── 📂 models/
│       ├── final_model.pkl                ← Stacking ensemble (trained)
│       └── scaler.pkl                     ← Fitted StandardScaler
│
├── 📂 streamlit_frontend/
│   ├── 🐍 app.py                          ← Streamlit dashboard
│   └── 📋 requirements.txt
│
├── 📂 src/
│   ├── 🐍 data_preprocessing.py           ← Load, clean, scale, SMOTE
│   ├── 🐍 train_base_models.py            ← RF, XGBoost, LightGBM + Optuna
│   ├── 🐍 stacking_optuna_meta_only.py    ← Stacking ensemble training
│   └── 🐍 evaluate.py                     ← Evaluation pipeline
│
├── 📂 models/
│   ├── rf_model.pkl                       ← Random Forest (Optuna tuned)
│   ├── xgb_model.pkl                      ← XGBoost (Optuna tuned)
│   ├── lgbm_model.pkl                     ← LightGBM (Optuna tuned)
│   ├── stacking_optuna_meta.pkl           ← Final stacking ensemble
│   ├── evaluation_results.csv             ← Model comparison table
│   ├── confusion_matrices.png             ← All model confusion matrices
│   ├── roc_curves.png                     ← ROC comparison chart
│   └── meta_importance.png                ← Meta learner feature importance
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── credit card clients.xlsx       ← UCI raw dataset
│   └── 📂 processed/
│       ├── X_train.pkl / y_train.pkl      ← SMOTE-balanced training data
│       ├── X_test.pkl / y_test.pkl        ← Held-out test data
│       └── scaler.pkl                     ← Fitted scaler
│
├── 📂 notebooks/
│   └── EDA.ipynb                          ← Exploratory data analysis
│
├── 🐳 docker-compose.yml
└── 📄 README.md
```

---

## 🎬 Recruiter Demo Guide

> **Show this in order for maximum impact. Each step builds on the previous.**

---

### 🥇 Step 1 — Lead with the Frontend (Best First Impression)

Open your Streamlit app. Show the dashboard header:

```
Credit Default Predictor
FastAPI + Docker + AWS EC2 + Streamlit
```

**Live demo — change values and predict:**
- Set Credit Limit: `20,000`
- Set Payment Status: `Delay 2 months`
- Set Last Bill: `18,000`
- Hit **Predict Default Risk**

Point out:
- Probability gauge goes red for high-risk customers
- Risk level: `HIGH / MEDIUM / LOW`
- Threshold slider adjusts sensitivity in real time

**Say:** *"This is a production ML app — predictions are served by a live REST API running on AWS EC2."*

---

### 🥈 Step 2 — Show the Live API (Swagger UI)

Open: [http://3.109.32.46:8000/docs](http://3.109.32.46:8000/docs)

Click `POST /api/v1/predict` → **Try it out** → paste this and Execute:

```json
{
  "features": [50000, 1, 2, 1, 35, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 5000, 0, 0, 0, 0, 0],
  "threshold": 0.3
}
```

**Say:** *"This is the FastAPI backend — auto-documented with Swagger. Recruiters can test the API directly in the browser without any setup."*

---

### 🥉 Step 3 — Show Health Monitoring

Open: [http://3.109.32.46:8000/health](http://3.109.32.46:8000/health)

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Say:** *"Health endpoints are a standard MLOps practice — used by DevOps teams and load balancers to monitor service availability."*

---

### 4️⃣ Step 4 — Show Docker (Proves Deployment Skills)

Open a terminal. Run:

```bash
sudo docker ps
```

Show the running container with port `0.0.0.0:8000->8000/tcp`.

Then show the public image:

```bash
docker pull vikash4122002/credit-default-api:latest
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest
```

**Say:** *"The entire FastAPI app is containerized. Anyone can pull this image and run the API in one command — no environment setup needed."*

---

### 5️⃣ Step 5 — Show AWS EC2 (Proves Cloud Skills)

Open AWS Console → EC2 → Running Instances.

Show:
- Instance state: **Running**
- Public IPv4: `3.109.32.46`
- Security group: Port 8000 open to `0.0.0.0/0`

**Say:** *"This is deployed on AWS EC2. The API is globally accessible — anyone with the IP can call it. This is how real ML services are hosted in production."*

---

### 6️⃣ Step 6 — Show the GitHub Repository

Walk through the structure:
- `src/` — clean, modular ML pipeline
- `deployment_api/` — FastAPI backend with proper routing
- `streamlit_frontend/` — interactive UI
- `Dockerfile` — containerization
- `README.md` — professional documentation

**Say:** *"The project is production-structured — separate concerns for data, training, serving, and frontend."*

---

### 7️⃣ Step 7 — Explain the ML Architecture

Draw or point to this:

```
Random Forest ──┐
XGBoost       ──┼──► Meta XGBoost ──► Final Prediction
LightGBM      ──┘
```

**Say:** *"I used a Stacking Ensemble — three base models trained with Optuna hyperparameter tuning, whose probability outputs become features for a meta XGBoost learner. This outperforms any single model."*

---

### 8️⃣ Step 8 — Explain the Business Context

**Say:** *"In credit risk, false negatives — missing a defaulter — are far more expensive than false positives. So I tuned the classification threshold to 0.30 to maximize recall (57.1%), prioritizing catching actual defaulters over precision."*

---

### ✅ Closing Statement

> *"This project demonstrates complete end-to-end ML engineering — from data preprocessing and ensemble model training with Optuna, to FastAPI backend development, Docker containerization, AWS EC2 cloud deployment, and a Streamlit frontend. It's production-ready and publicly accessible right now."*

---

### 💼 What Recruiters See

| Skill | Evidence |
|---|---|
| ✅ Machine Learning | Stacking Ensemble, SMOTE, threshold tuning |
| ✅ Model Optimization | Optuna Bayesian tuning (90 total trials) |
| ✅ Backend Development | FastAPI REST API with Pydantic validation |
| ✅ API Documentation | Swagger UI at /docs |
| ✅ Docker | Containerized app on Docker Hub |
| ✅ AWS Cloud | Live EC2 deployment |
| ✅ Frontend | Streamlit dashboard with Plotly |
| ✅ MLOps | Health monitoring, batch prediction, model serving |
| ✅ Software Engineering | Modular codebase, clean project structure |
| ✅ Business Thinking | Recall-focused for credit risk domain |

---

## ⚙️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System.git
cd AI-Powered-Credit-Default-Prediction-System
```

### Option A — Run with Docker (Recommended)

```bash
docker pull vikash4122002/credit-default-api:latest
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest
# Open: http://localhost:8000/docs
```

### Option B — Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (if running from scratch)
python src/data_preprocessing.py
python src/train_base_models.py
python src/stacking_optuna_meta_only.py

# Start API
cd deployment_api
uvicorn app:app --host 0.0.0.0 --port 8000

# Start Frontend (new terminal)
cd streamlit_frontend
streamlit run app.py
```

---

## 🚀 Future Enhancements

- [ ] **JWT Authentication** — Secure API with token-based auth
- [ ] **Database Integration** — Store predictions in PostgreSQL
- [ ] **CI/CD Pipeline** — GitHub Actions for auto-deploy on push
- [ ] **Kubernetes** — Orchestrate containers at scale
- [ ] **Model Monitoring** — Track drift with Evidently AI
- [ ] **SHAP Explainability** — Per-prediction feature importance
- [ ] **A/B Testing** — Compare model versions in production
- [ ] **Cloud Load Balancer** — AWS ALB for high availability

---

## 👨‍💻 Author

<div align="center">

**Vikash Kumar**

B.Tech | Machine Learning | FastAPI | Docker | AWS | Streamlit | MLOps

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vikash4122002)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vikash4122002/credit-default-api)

</div>

---

<div align="center">

### ⭐ If this project helped you, give it a star!

*Built with real engineering, real deployment, real business thinking.*

**Keywords:** `Machine Learning` `FastAPI` `Docker` `AWS EC2` `Streamlit` `MLOps` `REST API` `Ensemble Learning` `Credit Risk` `XGBoost` `LightGBM` `Stacking` `Optuna` `SMOTE` `Production ML`

</div>
