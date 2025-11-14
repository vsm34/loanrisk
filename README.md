#  LoanRisk – Credit Risk Scoring API

**Live API:** [https://loanrisk.onrender.com](https://loanrisk.onrender.com)  
**Demo:** [https://loanrisk.onrender.com/demo](https://loanrisk.onrender.com/demo)  
**Docs (Swagger):** [https://loanrisk.onrender.com/docs](https://loanrisk.onrender.com/docs)  
**Repository:** [https://github.com/vsm34/loanrisk](https://github.com/vsm34/loanrisk)

---

##  Overview

**LoanRisk** is a **cloud-hosted, production-style credit risk scoring microservice**.  
It ingests applicant data (income, employment, credit history, loan terms, etc.),  
predicts the probability of default, and returns:

-  **Approval / Review decision**
-  **Calibrated probability of default**
-  **Top explanatory factors** (using SHAP / linear fallback)
-  **Structured logging** of every request for analytics

This project simulates a real-world fintech underwriting API — built with  
**FastAPI**, **scikit-learn**, **Docker**, and deployed free-tier on **Render**.

---

##  Features

| Category | Description |
|-----------|-------------|
| **Machine Learning** | Logistic Regression pipeline trained on Kaggle’s Loan Prediction dataset |
| **Endpoints** | `/api/v1/score`, `/api/v1/batch/score`, `/api/v1/models/active`, `/api/v1/health` |
| **Explainability** | SHAP permutation explainer (with linear fallback) |
| **Validation** | Pydantic schemas with clear error handling |
| **Security** | Simple API key (`x-api-key`) header |
| **Logging** | SQLite table `request_logs` with latency, decision, and status |
| **Deployment** | Docker container running on Render (Free Plan) |
| **Demo** | Public `/demo` form with live model inference |

---

##  System Architecture

Client (Browser)
│
▼
FastAPI Backend
├─ /api/v1/score → Single applicant scoring (with API key)
├─ /api/v1/batch/score → Batch CSV/array scoring
├─ /api/v1/models/active → Model metadata manifest
├─ /api/v1/health → Health + version check
├─ /demo → Interactive human demo (HTML/JS)
└─ /demo/score → Server-side scoring for demo

## yaml

The backend loads the latest trained model from `/model/artifacts/`  
and performs real-time predictions using the same feature engineering  
pipeline applied during training.

---

## 🛠 Quick Start (Local)

### 1️⃣ Python Virtual Environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python model/train.py
uvicorn api.main:app --reload --port 8080
```
Open:

http://127.0.0.1:8080/demo → Live demo

http://127.0.0.1:8080/docs → Swagger docs

### 2️⃣ Docker (Recommended)
```bash
docker build -t loanrisk .
docker run -p 8090:8080 -e API_KEY=devkey --name loanrisk_api loanrisk
```
Then visit:
- http://127.0.0.1:8090/demo
- http://127.0.0.1:8090/docs

##  Environment Variables
- Variable	Default	Description
- API_KEY	devkey	API authentication key
- DB_URL	sqlite:///loanlens.db	Database URL for request logs

##  API Usage Example
POST /api/v1/score
```bash 
x-api-key: devkey
Content-Type: application/json
```
Body

json
{
  "payload": {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }
}
Response

json
Copy code
{
  "model_version": "1.0.0",
  "prob_default": 0.137,
  "decision": "approve",
  "threshold": 0.25,
  "explanations": [
    {"feature": "Credit_History", "impact": "risk_reducing"},
    {"feature": "LoanAmount", "impact": "high_risk"},
    {"feature": "Education_Graduate", "impact": "risk_reducing"}
  ],
  "confidence": 0.8,
  "request_id": "b8a2c4"
}

##  Logging (SQLite Schema)
All requests are automatically logged to loanlens.db:

- Column	Type	Description
- id	int	Auto ID
- ts	text	Timestamp
- request_id	text	Unique request key
- model_version	text	Active model version
- latency_ms	real	Response time
- decision	text	Approve / Review
- prob_default	real	Predicted risk
- endpoint	text	API endpoint hit
- status_code	int	HTTP response code
- client_api_key	text	Partial masked key

## Project Tech Stack
- Backend: FastAPI (Python 3.11)
- ML/Explainability: scikit-learn, pandas, numpy, SHAP
- Storage: SQLite (via SQLAlchemy)
- Deployment: Docker + Render (Free Plan)
- CI/CD: GitHub + Render Auto Deploy
- Docs/UI: FastAPI Swagger (/docs) + Custom HTML Demo (/demo)

 ## Model Training Summary
- Dataset: Kaggle Loan Prediction
- Algorithm: Logistic Regression
- Feature Engineering: One-hot encoding, income-debt ratios, scaling
- Metrics:
   - ROC-AUC ≈ 0.85
   - PR-AUC ≈ 0.90
   - Brier ≈ 0.14
   - KS Statistic ≈ 0.63

Artifacts saved under /model/artifacts/ as .joblib and .json.

## Deployment Steps (Render)
- Push repo to GitHub
- Connect GitHub → Render → New Web Service
- Runtime: Docker
- Environment Variable: API_KEY=devkey
- Instance: Free plan
- Port: (default 8080 from Dockerfile)

## After deploy:
- https://loanrisk.onrender.com
- /demo for users
- /docs for developers

 ## Summary
- ML model	        ✅ Trained
- API endpoints	    ✅ Live
- Cloud deployment	✅ Render Free Tier
- Demo UI	        ✅ Interactive
- Logging	        ✅ SQLite integrated
- Docs	            ✅ Auto-generated