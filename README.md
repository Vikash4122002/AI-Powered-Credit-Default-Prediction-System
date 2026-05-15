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

[![Live API](https://img.shields.io/badge/🚀_Live_API-Online-brightgreen?style=for-the-badge)](http://3.109.32.46:8000/docs)

[![Swagger Docs](https://img.shields.io/badge/📖_Swagger_Docs-Available-blue?style=for-the-badge)](http://3.109.32.46:8000/docs)

[![Health Check](https://img.shields.io/badge/💚_Health_Check-Healthy-success?style=for-the-badge)](http://3.109.32.46:8000/health)

[![Docker Hub](https://img.shields.io/badge/🐳_Docker_Hub-vikash4122002-2496ED?style=for-the-badge)](https://hub.docker.com/r/vikash4122002/credit-default-api)

</div>

---

# 📌 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🔗 Live Demo Links](#-live-demo-links)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 Machine Learning Pipeline](#-machine-learning-pipeline)
- [📊 Model Performance](#-model-performance)
- [🔌 API Endpoints](#-api-endpoints)
- [📘 Feature Reference](#-feature-reference)
- [🐳 Docker Deployment](#-docker-deployment)
- [☁️ AWS EC2 Deployment](#️-aws-ec2-deployment)
- [🖥️ Streamlit Frontend](#️-streamlit-frontend)
- [💡 Business Context](#-business-context)
- [🛠️ Tech Stack & Skills Used](#️-tech-stack--skills-used)
- [📁 Project Structure](#-project-structure)
- [🎬 Recruiter Demo Guide](#-recruiter-demo-guide)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Future Enhancements](#-future-enhancements)
- [👨‍💻 Author](#-author)

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

## 🔌 API Endpoints

> 🌐 **Live API Base URL:** `http://3.109.32.46:8000`

---

### 💚 1. Health Check Endpoint

```http
GET http://3.109.32.46:8000/health
```

#### Sample Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 📖 2. Interactive Swagger Documentation

```http
GET http://3.109.32.46:8000/docs
```

> Automatically generated FastAPI Swagger UI for testing API endpoints directly in the browser.

---

### 🎯 3. Single Prediction Endpoint

```http
POST http://3.109.32.46:8000/api/v1/predict
Content-Type: application/json
```

#### Sample Request Body

```json
{
  "features": [
    50000,
    1,
    2,
    1,
    35,
    0,
    0,
    0,
    0,
    0,
    0,
    10000,
    0,
    0,
    0,
    0,
    0,
    5000,
    0,
    0,
    0,
    0,
    0
  ],
  "threshold": 0.3
}
```

#### Sample Response

```json
{
  "prediction": 0,
  "probability": 0.187,
  "risk_level": "LOW",
  "threshold_used": 0.3
}
```

---

### 📦 4. Batch Prediction Endpoint

```http
POST http://3.109.32.46:8000/api/v1/predict_batch
Content-Type: application/json
```

#### Sample Request Body

```json
{
  "customers": [
    [
      50000, 1, 2, 1, 35,
      0, 0, 0, 0, 0, 0,
      10000, 0, 0, 0, 0, 0,
      5000, 0, 0, 0, 0, 0
    ],
    [
      20000, 2, 3, 2, 28,
      2, 2, 0, 0, 0, 0,
      15000, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0
    ]
  ],
  "threshold": 0.3
}
```

---

## 📘 Feature Reference

### Input Features Used for Prediction (23 Features)

| # | Feature | Description |
|---|---|---|
| 1 | LIMIT_BAL | Credit limit amount (NT dollars) |
| 2 | SEX | Gender (`1 = Male`, `2 = Female`) |
| 3 | EDUCATION | Education level (`1 = Graduate School`, `2 = University`, `3 = High School`, `4 = Others`) |
| 4 | MARRIAGE | Marital status (`1 = Married`, `2 = Single`, `3 = Others`) |
| 5 | AGE | Age of the customer |
| 6–11 | PAY_0 to PAY_6 | Monthly repayment status history |
| 12–17 | BILL_AMT1 to BILL_AMT6 | Monthly bill statement amounts |
| 18–23 | PAY_AMT1 to PAY_AMT6 | Previous monthly payment amounts |

> 📌 Repayment status values:
>
> - `-2` → No consumption
> - `-1` → Paid duly
> - `1–9` → Number of months payment delayed

---

## 🐳 Docker Deployment

### 🚀 Option A — Pull from Docker Hub (Recommended)

```bash
# Pull Docker image
docker pull vikash4122002/credit-default-api:latest

# Run Docker container
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest

# Verify running containers
docker ps

# Test API health endpoint
curl http://localhost:8000/health
```

---

### 🛠️ Option B — Build Docker Image Locally

```bash
# Clone repository
git clone https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System.git

# Move into deployment directory
cd AI-Powered-Credit-Default-Prediction-System/deployment_api

# Build Docker image
docker build -t credit-default-api .

# Run Docker container
docker run -d -p 8000:8000 credit-default-api

# View running logs
docker logs $(docker ps -q --filter "ancestor=credit-default-api")
```

---

### 📦 Dockerfile Overview

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

### 🖥️ EC2 Instance Configuration

| Setting | Value |
|---|---|
| Instance Type | t2.micro (AWS Free Tier) |
| Operating System | Ubuntu 22.04 LTS |
| Open Port | 8000 |
| AWS Region | ap-south-1 (Mumbai) |
| Deployment Type | Dockerized FastAPI Application |

---

### 🚀 Deployment Steps on AWS EC2

```bash
# 1. Connect to EC2 instance
ssh -i "your-key.pem" ubuntu@3.109.32.46

# 2. Update packages
sudo apt-get update

# 3. Install Docker
sudo apt-get install -y docker.io

# 4. Start Docker service
sudo systemctl start docker

# 5. Enable Docker on boot
sudo systemctl enable docker

# 6. Pull Docker image
sudo docker pull vikash4122002/credit-default-api:latest

# 7. Run container with auto restart
sudo docker run -d --restart=always -p 8000:8000 vikash4122002/credit-default-api:latest

# 8. Verify running container
sudo docker ps

# 9. Test API health endpoint
curl http://localhost:8000/health
```

> 📌 AWS Security Group must allow inbound traffic on:
>
> - Port `8000` → FastAPI Backend
> - Port `8501` → Streamlit Frontend (optional)

---

> ⚠️ **IP Stability Tip:**  
> AWS EC2 public IP addresses may change when the instance is restarted.  
> To keep a permanent public URL, attach an **Elastic IP** from the AWS Console.

---

## 🖥️ Streamlit Frontend

### 🚀 Frontend Features

- Real-time single customer default prediction
- Batch prediction using CSV upload
- Adjustable risk threshold slider
- Interactive analytics dashboard
- API documentation tab integrated into frontend
- Live API status monitoring
- Download prediction results as CSV

---

### ▶️ Run Streamlit Frontend Locally

```bash
# Move to frontend directory
cd streamlit_frontend

# Install frontend dependencies
pip install -r requirements.txt

# Start Streamlit application
streamlit run app.py
```

---

### 📊 Frontend Capabilities

```text
✅ Real-time prediction using FastAPI backend
✅ Interactive Plotly probability gauge
✅ Risk classification: LOW / MEDIUM / HIGH
✅ Batch CSV upload & prediction
✅ Downloadable prediction reports
✅ Live API health status monitoring
✅ Responsive dashboard layout
✅ Integrated API documentation
```

---

## 💡 Business Context

### 📌 Problem Statement

Credit default prediction is a major challenge for banks and financial institutions.

Traditional rule-based systems often fail to detect complex financial behavior patterns and hidden risk signals.

This project demonstrates how machine learning can improve financial risk assessment by:

- Identifying high-risk customers earlier
- Reducing missed default cases
- Supporting large-scale batch prediction workflows
- Providing actionable risk-level insights

---

### 🎯 Why Recall Is Prioritized

In credit risk systems:

```text
False Negative:
A risky customer is classified as safe
→ Potential financial loss

False Positive:
A safe customer is flagged as risky
→ Manual review required
```

Because missing an actual defaulter is far more expensive, the model is intentionally optimized for **higher recall** instead of only maximizing accuracy.

This reflects real-world financial risk management strategy.

---

## 🛠️ Tech Stack & Skills Used

| Category | Technology | Purpose |
|---|---|---|
| Machine Learning | Scikit-learn | Model training pipeline & evaluation |
| Boosting Models | XGBoost | Base model & meta learner |
| Boosting Models | LightGBM | Base ensemble model |
| Ensemble Learning | StackingClassifier | Multi-model ensemble |
| Hyperparameter Tuning | Optuna | Bayesian optimization |
| Imbalanced Learning | SMOTE | Minority class balancing |
| Feature Scaling | StandardScaler | Data normalization |
| Backend API | FastAPI + Uvicorn | REST API deployment |
| Validation | Pydantic | Request validation |
| Frontend | Streamlit | Interactive dashboard |
| Visualization | Plotly | Interactive charts & gauges |
| Containerization | Docker | Application packaging |
| Container Registry | Docker Hub | Public Docker image hosting |
| Cloud Deployment | AWS EC2 | Cloud hosting |
| Data Processing | Pandas + NumPy | Data handling |
| Evaluation | Matplotlib + Seaborn | Performance visualization |
| Model Persistence | Joblib | Saving trained models |
| Version Control | Git + GitHub | Source control & collaboration |

---

## 📁 Project Structure

```text
📦 AI-Powered-Credit-Default-Prediction-System
│
├── 📂 deployment_api/
│   ├── 🐍 app.py
│   ├── 🐳 Dockerfile
│   ├── 📋 requirements.txt
│   ├── 📂 routes/
│   ├── 📂 services/
│   ├── 📂 schema/
│   ├── 📂 utils/
│   └── 📂 models/
│       └── ⚠️ Trained model files bundled inside Docker image
│
├── 📂 streamlit_frontend/
│   ├── 🐍 app.py
│   └── 📋 requirements.txt
│
├── 📂 src/
│   ├── 🐍 data_preprocessing.py
│   ├── 🐍 train_base_models.py
│   ├── 🐍 stacking_optuna_meta_only.py
│   └── 🐍 evaluate.py
│
├── 📂 models/
│   ├── evaluation_results.csv
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── meta_importance.png
│
│   ⚠️ Trained .pkl model files are excluded from GitHub
│   due to GitHub file size limitations.
│
├── 📂 screenshots/
│   ├── streamlit_dashboard.png
│   ├── swagger_docs.png
│   ├── docker_running.png
│   ├── aws_ec2.png
│   ├── roc_curve.png
│   └── confusion_matrix.png
│
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```
---

## 🎬 Recruiter Demo Guide

> 📌 Recommended order for demonstrating the project during interviews or recruiter discussions.

---

### 🥇 Step 1 — Demonstrate the Streamlit Frontend

Open the Streamlit dashboard and perform a live prediction.

#### Example Input

- Credit Limit: `20,000`
- Payment Delay Status: `2 months delayed`
- Latest Bill Amount: `18,000`

Click **Predict Default Risk**.

#### Highlight During Demo

- Real-time prediction
- Probability gauge visualization
- Risk level classification (`LOW`, `MEDIUM`, `HIGH`)
- Adjustable prediction threshold slider

#### Explain

> “This frontend communicates with a live FastAPI backend deployed on AWS EC2. Predictions are generated in real time using a stacking ensemble machine learning model.”

---

### 🥈 Step 2 — Show the Live FastAPI Swagger Documentation

Open:

```text
http://3.109.32.46:8000/docs
```

Use the interactive Swagger UI to test the prediction API directly in the browser.

#### Example Request

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

#### Explain

> “FastAPI automatically generates interactive API documentation using Swagger UI, making testing and integration easier for developers.”

---

### 🥉 Step 3 — Demonstrate Health Monitoring

Open:

```text
http://3.109.32.46:8000/health
```

#### Sample Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Explain

> “Health monitoring endpoints are commonly used in production systems for service monitoring, load balancers, and uptime checks.”

---

### 4️⃣ Step 4 — Demonstrate Docker Deployment

Run:

```bash
sudo docker ps
```

Show the running FastAPI container.

Then demonstrate pulling the public Docker image:

```bash
docker pull vikash4122002/credit-default-api:latest
```

#### Explain

> “The entire application is containerized using Docker, allowing consistent deployment across environments.”

---

### 5️⃣ Step 5 — Demonstrate AWS EC2 Deployment

Open the AWS Console and show:

- Running EC2 instance
- Public IP address
- Open security group ports
- Dockerized backend deployment

#### Explain

> “The machine learning API is deployed on AWS EC2 and accessible publicly through REST endpoints.”

---

### 6️⃣ Step 6 — Walk Through the GitHub Repository

Highlight the following directories:

| Folder | Purpose |
|---|---|
| `src/` | Machine learning training pipeline |
| `deployment_api/` | FastAPI backend service |
| `streamlit_frontend/` | Interactive dashboard frontend |
| `models/` | Evaluation graphs and metrics |
| `README.md` | Full project documentation |

#### Explain

> “The project follows a modular software engineering structure separating training, deployment, frontend, and evaluation components.”

---

### 7️⃣ Step 7 — Explain the Ensemble ML Architecture

```text
Random Forest ──┐
XGBoost       ──┼──► Meta XGBoost ──► Final Prediction
LightGBM      ──┘
```

#### Explain

> “Three optimized base models generate probability predictions which are combined using a meta XGBoost learner through stacking ensemble learning.”

---

### 8️⃣ Step 8 — Explain the Business Logic

#### Explain

> “The classification threshold was intentionally reduced to 0.30 to maximize recall because missing a real defaulter creates higher financial risk than incorrectly flagging a safe customer.”

---

## ✅ Final Closing Statement

> “This project demonstrates the complete machine learning engineering lifecycle including data preprocessing, ensemble learning, Optuna hyperparameter optimization, FastAPI backend development, Docker containerization, AWS EC2 cloud deployment, and interactive Streamlit frontend integration.”

---


---

## 🚀 Key Skills Demonstrated

| Skill Area | Demonstrated Through |
|---|---|
| ✅ Machine Learning | Stacking Ensemble, SMOTE, Threshold Optimization |
| ✅ Hyperparameter Tuning | Optuna Bayesian Optimization |
| ✅ Ensemble Learning | Random Forest + XGBoost + LightGBM |
| ✅ Backend Development | FastAPI REST API with Pydantic Validation |
| ✅ API Documentation | Auto-generated Swagger UI |
| ✅ Docker | Fully Containerized Application |
| ✅ Cloud Deployment | AWS EC2 Live Deployment |
| ✅ Frontend Development | Streamlit Interactive Dashboard |
| ✅ MLOps Concepts | Health Monitoring, Batch Prediction, Model Serving |
| ✅ Software Engineering | Modular Project Structure & Clean Code |
| ✅ Business Understanding | Recall-focused optimization for credit risk |

---

## ⚙️ Installation & Setup

### 📥 Clone the Repository

```bash
git clone https://github.com/vikash4122002/AI-Powered-Credit-Default-Prediction-System.git

cd AI-Powered-Credit-Default-Prediction-System
```

---

### 🐳 Option A — Run with Docker (Recommended)

```bash
# Pull Docker image
docker pull vikash4122002/credit-default-api:latest

# Run Docker container
docker run -d -p 8000:8000 vikash4122002/credit-default-api:latest

# Open Swagger API documentation
http://localhost:8000/docs
```

---

### 🛠️ Option B — Train & Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from UCI repository
# Save dataset inside:
# data/raw/credit card clients.xlsx

# Run preprocessing pipeline
python src/data_preprocessing.py

# Train base models
python src/train_base_models.py

# Train stacking ensemble
python src/stacking_optuna_meta_only.py

# Start FastAPI backend
cd deployment_api

uvicorn app:app --host 0.0.0.0 --port 8000

# Start Streamlit frontend (new terminal)
cd streamlit_frontend

streamlit run app.py
```

---

## 🚀 Future Enhancements

- [ ] JWT Authentication for API security
- [ ] PostgreSQL database integration
- [ ] CI/CD pipeline using GitHub Actions
- [ ] Kubernetes deployment support
- [ ] Model drift monitoring using Evidently AI
- [ ] SHAP explainability integration
- [ ] Elastic IP for stable AWS deployment
- [ ] A/B testing framework for model comparison

---

## 👨‍💻 Author

<div align="center">

## Vikash Kumar

B.Tech ECE · Machine Learning · FastAPI · Docker · AWS · Streamlit · MLOps

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vikash4122002)

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vikash4122002/credit-default-api)

</div>

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

Built with production-oriented engineering, cloud deployment, and real-world ML workflow design.

</div>

---

## 🔑 Keywords

`Machine Learning` · `FastAPI` · `Docker` · `AWS EC2` · `Streamlit` · `MLOps` · `REST API` · `Ensemble Learning` · `XGBoost` · `LightGBM` · `Stacking` · `Optuna` · `SMOTE` · `Credit Risk Prediction`

---
