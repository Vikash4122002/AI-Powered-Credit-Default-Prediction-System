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
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **30,000 credit card clients · Stacking Ensemble ML · FastAPI REST API · Dockerized · Deployed on AWS EC2 · 76.2% ROC-AUC**

<br/>

[![Live API](https://img.shields.io/badge/🚀_Live_API-EC2_Deployed-brightgreen?style=for-the-badge)](#-aws-ec2-deployment)
[![Swagger Docs](https://img.shields.io/badge/📖_Swagger_Docs-Available-blue?style=for-the-badge)](#2-interactive-swagger-docs)
[![Health Check](https://img.shields.io/badge/💚_Health_Check-Online-success?style=for-the-badge)](#1-health-check)
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
- [Screenshots](#-screenshots)
- [API Endpoints](#-api-endpoints)
- [Docker Deployment](#-docker-deployment)
- [AWS EC2 Deployment](#-aws-ec2-deployment)
- [Streamlit Frontend](#-streamlit-frontend)
- [Business Context](#-business-context)
- [Tech Stack & Skills Used](#-tech-stack--skills-used)
- [Project Structure](#-project-structure)
- [Recruiter Demo Guide](#-recruiter-demo-guide)
- [Installation & Setup](#-installation--setup)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Project Overview

This is an **end-to-end deployed ML project** that predicts credit card default risk using an advanced Stacking Ensemble combining XGBoost, LightGBM, and Random Forest. The trained model is served via a **FastAPI REST API**, containerized with **Docker**, deployed on **AWS EC2**, and accessible through an interactive **Streamlit frontend**.

This project covers a real-world financial risk workflow — going from raw data all the way to a live, accessible prediction API.

> Built and deployed an end-to-end machine learning system for credit default prediction using Stacking Ensemble models, FastAPI, Docker, AWS EC2, and Streamlit.

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

Built with production-oriented engineering, cloud deployment, and real-world ML workflow design.

---

## 🔗 Live Demo Links

> ⚠️ **Note:** The API is deployed on AWS EC2.  
> The current public IP is: `3.109.32.46`  
> If the EC2 instance is restarted, the public IP may change unless an **Elastic IP** is attached.

| Resource | URL |
|---|---|
| 🚀 **Live API Base** | http://3.109.32.46:8000 |
| 📖 **Swagger UI (Interactive Docs)** | http://3.109.32.46:8000/docs |
| 💚 **Health Monitoring Endpoint** | http://3.109.32.46:8000/health |
| 🐳 **Docker Hub Image** | `docker pull vikash4122002/credit-default-api:latest` |
| 💻 **GitHub Repository** | https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System |

---

## 🏆 Key Results at a Glance

<div align="center">

| Metric | Value | Notes |
|---|---|---|
| 🧠 **Best Model** | Stacking Ensemble | RF + XGBoost + LightGBM + Meta XGBoost |
| 📈 **ROC-AUC** | **76.2%** | Key metric for imbalanced classification |
| 🎯 **Recall** | **57.1%** | Most critical — catching actual defaulters |
| ⚡ **Precision** | **46.8%** | Balancing false positives |
| 📊 **F1 Score** | **51.4%** | Harmonic mean of precision & recall |
| 🔢 **Accuracy** | **76.2%** | At threshold = 0.30 |
| ⚙️ **Threshold** | **0.30** | Custom tuned to maximize recall |
| 📦 **Deployment** | **AWS EC2 + Docker** | Live, accessible API |
| 🔄 **SMOTE** | Applied | Balanced class imbalance on training set only |
| 🔍 **Tuning** | **Optuna (90 trials)** | 30 trials per base model + 20 for meta learner |

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
│      Public IP: <EC2-PUBLIC-IP>  |  Port: 8000          │
│              Globally Accessible API                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Pipeline

### Step 1 — Data Preprocessing

- **Dataset:** UCI Credit Card Default Dataset containing 30,000 customer records and 23 financial features
- **Data Cleaning:** Corrected invalid categorical values in `EDUCATION` and `MARRIAGE`
- **Missing Value Handling:** Filled missing values using median imputation
- **Duplicate Removal:** Removed duplicate customer records
- **Train-Test Split:** Applied stratified 80/20 split to preserve class distribution
- **Feature Scaling:** Used `StandardScaler` fitted only on training data to prevent data leakage
- **Class Imbalance Handling:** Applied SMOTE only on the training dataset to balance minority default cases

---

### Step 2 — Hyperparameter Optimization with Optuna

| Model | Optimization Method | Important Tuned Parameters |
|---|---|---|
| 🌲 Random Forest | Optuna Bayesian Optimization | n_estimators, max_depth, min_samples_split, max_features |
| ⚡ XGBoost | Optuna Bayesian Optimization | learning_rate, max_depth, subsample, colsample_bytree |
| 🔦 LightGBM | Optuna Bayesian Optimization | num_leaves, learning_rate, subsample, colsample_bytree |

- Total optimization trials performed: **90+**
- Used Optuna TPE sampler for efficient hyperparameter search

---

### Step 3 — Stacking Ensemble Architecture

```text
Base Models (Level 0)
│
├── Random Forest
├── XGBoost
└── LightGBM
        │
        ▼
Probability outputs used as meta-features
        │
        ▼
Meta Learner (Level 1)
└── XGBoost (Optimized with Optuna)
        │
        ▼
Final Credit Default Prediction
```

#### Ensemble Configuration

- **Cross Validation:** 5-Fold StratifiedKFold
- **Stack Method:** `predict_proba`
- **Passthrough:** Disabled
- **Meta Learner:** Optimized XGBoost Classifier

---

### Step 4 — Threshold Optimization

Instead of using the default classification threshold of `0.50`, a custom threshold optimization strategy was implemented.

#### Optimization Formula

```python
score = 0.7 * recall + 0.3 * precision
```

#### Why Threshold = 0.30?

In credit risk prediction:

- Missing a real defaulter (**False Negative**) can lead to major financial losses
- Incorrectly flagging a safe customer (**False Positive**) is comparatively less costly

Therefore, the system prioritizes **Recall** to detect as many risky customers as possible.

This threshold was selected to maximize business impact rather than simply maximizing accuracy.

---

## 📊 Model Performance

### Final Model Comparison

| Model | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|
| 🥇 Stacking Ensemble | **57.1%** | **46.8%** | **51.4%** | **76.2%** |
| ⚡ XGBoost | 55.8% | 47.2% | 51.1% | 75.8% |
| 🔦 LightGBM | 54.9% | 47.5% | 51.0% | 75.5% |
| 🌲 Random Forest | 52.3% | 48.1% | 50.1% | 74.9% |

---

## 🎯 Why Recall Is the Most Important Metric

```text
Business Scenario: Credit Risk Prediction

False Negative:
A risky customer is predicted as safe
→ Potential financial loss for the bank

False Positive:
A safe customer is predicted as risky
→ Manual review required

Cost of False Negative >>> Cost of False Positive
```

### Final Business Decision

The system was intentionally optimized for **high recall** to identify the maximum number of potential defaulters while maintaining acceptable precision.

This reflects a real-world financial risk management strategy used in banking and lending systems.

---

---

## 📷 Screenshots

### Streamlit Dashboard
> *Screenshot: Run the Streamlit app → take a screenshot → save as `screenshots/streamlit_dashboard.png`*

![Streamlit Dashboard](screenshots/streamlit_dashboard.png)

---

### Swagger API Docs
> *Screenshot: Open `http://<EC2-PUBLIC-IP>:8000/docs` → screenshot → save as `screenshots/swagger_docs.png`*

![Swagger Docs](screenshots/swagger_docs.png)

---

### Docker Running on EC2
> *Screenshot: Run `sudo docker ps` on EC2 terminal → screenshot → save as `screenshots/docker_running.png`*

![Docker Running](screenshots/docker_running.png)

---

### AWS EC2 Instance
> *Screenshot: AWS Console → EC2 → Running Instances → screenshot → save as `screenshots/aws_ec2.png`*

![AWS EC2](screenshots/aws_ec2.png)

> 💡 **To add screenshots:** Create a `screenshots/` folder in your repo root, add the four images above, and they will render automatically here.

## 🔌 API Endpoints

> Current AWS EC2 Public IP: `3.109.32.46`

### 1. Health Check

```http
GET http://3.109.32.46:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. Interactive Swagger Docs

```http
GET http://3.109.32.46:8000/docs
```

> Full interactive API — try predictions directly in the browser. No Postman or code required.

---

### 3. Single Prediction

```http
POST http://3.109.32.46:8000/api/v1/predict
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

---

### 4. Batch Prediction

```http
POST http://3.109.32.46:8000/api/v1/predict_batch
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

### Feature Reference (23 features in order)

| # | Feature | Description |
|---|---|---|
| 1 | LIMIT_BAL | Credit limit (NT dollars) |
| 2 | SEX | 1=Male, 2=Female |
| 3 | EDUCATION | 1=Graduate School, 2=University, 3=High School, 4=Others |
| 4 | MARRIAGE | 1=Married, 2=Single, 3=Others |
| 5 | AGE | Age in years |
| 6–11 | PAY_0 to PAY_6 | Repayment status (-2=no consumption, -1=paid duly, 1–9=months delayed) |
| 12–17 | BILL_AMT1–6 | Bill statement amounts (Apr–Sep 2005) |
| 18–23 | PAY_AMT1–6 | Previous payment amounts (Apr–Sep 2005) |

---

## 🐳 Docker Deployment

### Option A — Pull from Docker Hub (Fastest — no build needed)

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

### Option B — Build Locally

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
| **Instance Type** | t2.micro (Free Tier eligible) |
| **OS** | Ubuntu 22.04 LTS |
| **Port** | 8000 (open in Security Group inbound rules) |
| **Region** | ap-south-1 (Mumbai) |
| **Access** | Public IPv4 DNS |

### Deployment Steps on EC2

```bash
# 1. SSH into your EC2 instance
ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# 3. Pull the Docker image
sudo docker pull vikash4122002/credit-default-api:latest

# 4. Run with auto-restart so it survives reboots
sudo docker run -d --restart=always -p 8000:8000 vikash4122002/credit-default-api:latest

# 5. Verify deployment
sudo docker ps
curl http://localhost:8000/health
```

> ⚠️ **IP Stability Tip:** AWS EC2 public IPs change when an instance is stopped and restarted. To get a permanent URL, assign an **Elastic IP** to your instance in the AWS Console — it's free while the instance is running.

---

## 🖥️ Streamlit Frontend

### Features

- **Single Prediction Tab** — Enter customer details, adjust risk threshold, get instant prediction with probability gauge
- **Batch Prediction Tab** — Upload CSV, run bulk predictions, download results as CSV
- **Analytics Tab** — View model performance metrics and bar charts
- **API Docs Tab** — Integrated documentation with sample JSON requests

### Run Locally

```bash
cd streamlit_frontend
pip install -r requirements.txt
streamlit run app.py
```

### Frontend Capabilities

```
✅ Real-time prediction with Plotly probability gauge
✅ Risk level output: LOW / MEDIUM / HIGH
✅ Adjustable threshold slider (0.05 – 0.50)
✅ CSV batch upload & one-click download of results
✅ Live API status indicator (Online / Offline)
✅ Interactive Plotly visualizations
✅ Clean, wide-layout responsive design
```

---

## 💡 Business Context

### Problem Statement

Credit default is a significant risk for banks and financial institutions. Traditional rule-based systems often miss non-linear patterns in customer payment behaviour. This project demonstrates how machine learning can assist credit risk teams by:

- Identifying potentially high-risk customers earlier in the credit lifecycle
- Prioritising recall to reduce the number of missed defaulters
- Supporting batch processing for large-scale application review
- Providing clear, actionable risk level outputs (LOW / MEDIUM / HIGH) that credit officers can act on

> **Why recall over precision?** In credit risk, failing to flag a real defaulter (false negative) typically results in a larger financial exposure than incorrectly flagging a safe customer (false positive). The model threshold is tuned to reflect this real-world cost asymmetry.

---

## 🛠️ Tech Stack & Skills Used

| Category | Tool / Technology | Usage |
|---|---|---|
| **ML Framework** | Scikit-learn | Pipeline, stacking, cross-validation, metrics |
| **Boosting** | XGBoost | Base model + meta learner |
| **Boosting** | LightGBM | Base model |
| **Ensemble** | StackingClassifier | Combines all base models |
| **Tuning** | Optuna (TPE Sampler) | Bayesian hyperparameter optimization |
| **Imbalance** | SMOTE (imbalanced-learn) | Synthetic minority oversampling |
| **Scaling** | StandardScaler | Feature normalization (no leakage) |
| **Backend** | FastAPI + Uvicorn | REST API with async support |
| **Validation** | Pydantic | Request/response schema validation |
| **Frontend** | Streamlit | Interactive web dashboard |
| **Visualization** | Plotly | Gauge charts, histograms |
| **Containerization** | Docker | Full app containerization |
| **Registry** | Docker Hub | Public image hosting |
| **Cloud** | AWS EC2 | Cloud deployment |
| **Data** | Pandas, NumPy | Data processing & wrangling |
| **Evaluation** | Matplotlib, Seaborn | EDA and model evaluation plots |
| **Serialization** | Joblib | Model persistence (.pkl) |
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
│   │   └── model_service.py               ← Model loading & inference logic
│   ├── 📂 schema/
│   │   └── request_schema.py              ← Pydantic request/response models
│   ├── 📂 utils/
│   │   └── preprocessing.py               ← Input preprocessing helpers
│   └── 📂 models/
│       ├── final_model.pkl                ← Bundled inside Docker image
│       └── scaler.pkl                     ← Bundled inside Docker image
│
├── 📂 streamlit_frontend/
│   ├── 🐍 app.py                          ← Streamlit dashboard
│   └── 📋 requirements.txt
│
├── 📂 src/
│   ├── 🐍 data_preprocessing.py           ← Load, clean, scale, SMOTE
│   ├── 🐍 train_base_models.py            ← RF, XGBoost, LightGBM + Optuna
│   ├── 🐍 stacking_optuna_meta_only.py    ← Stacking ensemble training
│   └── 🐍 evaluate.py                     ← Full evaluation pipeline
│
├── 📂 models/
│   ├── evaluation_results.csv             ← Model comparison table
│   ├── confusion_matrices.png             ← All model confusion matrices
│   ├── roc_curves.png                     ← ROC comparison chart
│   └── meta_importance.png                ← Meta learner feature importance
│
│   ⚠️  Note: Trained .pkl model files are not committed to this repo due to
│   file size limits. To use the model: pull the Docker image (which bundles
│   the trained model), or re-train by running the scripts in src/ in order.
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── credit card clients.xlsx       ← UCI raw dataset
│   └── 📂 processed/                      ← Auto-generated by preprocessing script
│
├── 📂 notebooks/
│   └── EDA.ipynb                          ← Exploratory data analysis
│
├── 📂 screenshots/                        ← Add your project screenshots here
│   ├── streamlit_dashboard.png            ← Streamlit app screenshot
│   ├── swagger_docs.png                   ← Swagger UI screenshot
│   ├── docker_running.png                 ← docker ps terminal screenshot
│   └── aws_ec2.png                        ← AWS Console EC2 screenshot
│
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🎬 Recruiter Demo Guide

> **Follow this order for maximum impact. Each step builds on the previous one.**

---

### 🥇 Step 1 — Lead with the Frontend (Best First Impression)

Open your Streamlit app. Enter these values and predict live:

- Credit Limit: `20,000`
- Payment Status: `Delay 2 months`
- Last Bill: `18,000`
- Hit **Predict Default Risk**

Show the probability gauge going red. Point out the `HIGH` risk label and the adjustable threshold slider.

**Say:** *"This is a live ML app — predictions are served by a REST API running on AWS EC2."*

---

### 🥈 Step 2 — Show the Live API (Swagger UI)

Open: `http://<EC2-PUBLIC-IP>:8000/docs`

Click `POST /api/v1/predict` → **Try it out** → paste this → Execute:

```json
{
  "features": [50000, 1, 2, 1, 35, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 5000, 0, 0, 0, 0, 0],
  "threshold": 0.3
}
```

**Say:** *"This is the FastAPI backend with auto-generated Swagger docs. Anyone can test the API directly in the browser — no Postman or code needed."*

---

### 🥉 Step 3 — Show Health Monitoring

Open: `http://<EC2-PUBLIC-IP>:8000/health`

```json
{ "status": "healthy", "model_loaded": true }
```

**Say:** *"Health endpoints are standard practice in deployed ML services — used by DevOps and load balancers to monitor availability."*

---

### 4️⃣ Step 4 — Show Docker

```bash
sudo docker ps
```

Show the running container. Then demonstrate the public image:

```bash
docker pull vikash4122002/credit-default-api:latest
```

**Say:** *"The entire app is containerized. Anyone can pull this image and run the API locally in one command — no environment setup needed."*

---

### 5️⃣ Step 5 — Show AWS EC2

Open AWS Console → EC2 → Running Instances. Show the running instance, public IP, and Security Group with port 8000 open.

**Say:** *"This is deployed on AWS EC2. The API is globally accessible — this is how real ML services are hosted in the cloud."*

---

### 6️⃣ Step 6 — Walk Through the GitHub Repo

Show:
- `src/` — clean, modular ML training scripts
- `deployment_api/` — FastAPI backend with proper routing
- `streamlit_frontend/` — interactive UI
- `Dockerfile` — containerization config
- `README.md` — professional documentation

**Say:** *"The project is structured the way real ML teams organise their work — separate concerns for data, training, serving, and frontend."*

---

### 7️⃣ Step 7 — Explain the ML Architecture

```
Random Forest ──┐
XGBoost       ──┼──► Meta XGBoost ──► Final Prediction
LightGBM      ──┘
```

**Say:** *"I trained three base models, each tuned independently with Optuna. Their probability outputs become features for a meta XGBoost learner — this stacking approach consistently outperforms any single model."*

---

### 8️⃣ Step 8 — Explain the Business Reasoning

**Say:** *"In credit risk, missing a real defaulter costs far more than flagging a good customer. So I tuned the classification threshold to 0.30, which maximises recall at 57.1%. This is a deliberate, domain-aware engineering decision — not just a default setting."*

---

### ✅ Closing Statement

> *"This project covers the full ML engineering lifecycle — data preprocessing, ensemble model training with Optuna, FastAPI REST API, Docker containerization, AWS EC2 deployment, and a Streamlit frontend. The API is live and publicly accessible right now."*

---

### 💼 What Recruiters See

| Skill | Evidence |
|---|---|
| ✅ Machine Learning | Stacking Ensemble, SMOTE, threshold tuning |
| ✅ Model Optimization | Optuna Bayesian tuning (90 total trials) |
| ✅ Backend Development | FastAPI REST API with Pydantic validation |
| ✅ API Documentation | Swagger UI auto-generated at /docs |
| ✅ Docker | Containerized app published to Docker Hub |
| ✅ AWS Cloud | Live EC2 deployment, Security Group config |
| ✅ Frontend | Streamlit dashboard with Plotly gauge charts |
| ✅ MLOps | Health monitoring, batch prediction, model serving |
| ✅ Software Engineering | Modular codebase, clean project structure |
| ✅ Domain Thinking | Recall-first design with business justification |

---

## ⚙️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System.git
cd AI-Powered-Credit-Default-Prediction-System
```

### Option A — Run with Docker (Recommended — no training needed)

```bash
docker pull vikash4122002/credit-default-api:latest
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest
# Open: http://localhost:8000/docs
```

### Option B — Train from Scratch and Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
#    Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
#    Save to: data/raw/credit card clients.xlsx

# 3. Run the pipeline in order
python src/data_preprocessing.py
python src/train_base_models.py
python src/stacking_optuna_meta_only.py

# 4. Start the API
cd deployment_api
uvicorn app:app --host 0.0.0.0 --port 8000

# 5. Start the frontend (new terminal)
cd streamlit_frontend
streamlit run app.py
```

---

## 🚀 Future Enhancements

- [ ] **JWT Authentication** — Secure API with token-based auth
- [ ] **Database Integration** — Store predictions and audit trail in PostgreSQL
- [ ] **CI/CD Pipeline** — GitHub Actions for automated testing and deployment on push
- [ ] **Kubernetes** — Orchestrate containers at scale with auto-scaling
- [ ] **Model Monitoring** — Detect data drift with Evidently AI
- [ ] **SHAP Explainability** — Per-prediction feature importance for credit officers
- [ ] **Elastic IP** — Permanent AWS EC2 URL that survives instance restarts
- [ ] **A/B Testing Framework** — Compare model versions in production

---

## 👨‍💻 Author

<div align="center">

**Vikash Kumar**

B.Tech ECE · Machine Learning | FastAPI | Docker | AWS | Streamlit | MLOps

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vikash4122002)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR-LINKEDIN-ID)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vikash4122002/credit-default-api)

</div>

---

<div align="center">

### ⭐ If this project helped you, give it a star!

*Built with real engineering, real deployment, real domain thinking.*

**Keywords:** `Machine Learning` `FastAPI` `Docker` `AWS EC2` `Streamlit` `MLOps` `REST API` `Ensemble Learning` `Credit Risk` `XGBoost` `LightGBM` `Stacking` `Optuna` `SMOTE` `End-to-End ML`

</div>
