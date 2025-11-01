# api/main.py
import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from sqlalchemy import create_engine, text

APP_VERSION = "1.0.0"
API_KEY = os.getenv("API_KEY", "devkey")

# -------------------------------
# Model + manifest paths
# -------------------------------
ART_DIR = Path(__file__).parent.parent / "model" / "artifacts"
MODEL_PATH = ART_DIR / "model_v1_0_0.joblib"
MANIFEST_PATH = ART_DIR / "manifest_v1_0_0.json"

if not MODEL_PATH.exists():
    raise RuntimeError("Model artifact missing. Run: python model/train.py")

bundle = joblib.load(MODEL_PATH)
PIPE = bundle["pipeline"]
FEATURES = bundle["feature_cols"]

with open(MANIFEST_PATH) as f:
    MANIFEST = json.load(f)

# -------------------------------
# Database (SQLite) for request logs
# -------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///loanlens.db")
ENGINE = create_engine(DB_URL, future=True)

# Create table on startup if not exists
with ENGINE.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS request_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      request_id TEXT NOT NULL,
      model_version TEXT NOT NULL,
      latency_ms REAL NOT NULL,
      decision TEXT NOT NULL,
      prob_default REAL NOT NULL,
      status_code INTEGER NOT NULL,
      endpoint TEXT NOT NULL,
      client_api_key TEXT
    );
    """))

def _log_req(request_id: str,
             latency_ms: float,
             decision: str,
             prob_default: float,
             status_code: int,
             endpoint: str,
             api_key_value: str):
    """Write a single row into request_logs."""
    with ENGINE.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO request_logs
                (ts, request_id, model_version, latency_ms, decision,
                 prob_default, status_code, endpoint, client_api_key)
                VALUES (:ts, :rid, :ver, :lat, :dec, :prob, :code, :ep, :api)
            """),
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rid": request_id,
                "ver": MANIFEST["model_version"],
                "lat": float(latency_ms),
                "dec": decision,
                "prob": float(prob_default),
                "code": int(status_code),
                "ep": endpoint,
                # store a masked prefix for quick source grouping (avoid full secrets)
                "api": api_key_value[:4] + "****" if api_key_value else None
            }
        )

# -------------------------------
# FastAPI app + schemas
# -------------------------------
app = FastAPI(title="LoanLens API", version=APP_VERSION)

class ScoreRequest(BaseModel):
    payload: Dict[str, Any]

class BatchRequest(BaseModel):
    records: list[dict]

def _df_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """Coerce incoming payload dict to a DataFrame with the right feature order.
       Missing keys become None (imputed in the pipeline)."""
    row = {col: payload.get(col, None) for col in FEATURES}
    return pd.DataFrame([row])

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/api/v1/health")
def health():
    return {"status": "ok", "model_version": MANIFEST["model_version"]}

@app.get("/api/v1/models/active")
def active_model():
    return MANIFEST

@app.post("/api/v1/score")
def score(req: ScoreRequest, x_api_key: str = Header(default="")):
    t0 = time.time()

    # Auth
    if x_api_key != API_KEY:
        _log_req(uuid.uuid4().hex[:8], 0.0, "unauthorized", 0.0, 401, "/api/v1/score", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Prepare features
    df = _df_from_payload(req.payload)
    rid = uuid.uuid4().hex[:8]

    # Predict
    try:
        prob_default = float(PIPE.predict_proba(df)[0, 1])
    except Exception as e:
        _log_req(rid, (time.time() - t0) * 1000.0, "error", 0.0, 400, "/api/v1/score", x_api_key)
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")

    # Decision rule
    threshold = 0.25
    decision = "approve" if prob_default < threshold else "review"

    # Log + respond
    latency_ms = (time.time() - t0) * 1000.0
    _log_req(rid, latency_ms, decision, prob_default, 200, "/api/v1/score", x_api_key)

    return {
        "model_version": MANIFEST["model_version"],
        "prob_default": round(prob_default, 4),
        "decision": decision,
        "threshold": threshold,
        "explanations": [],  # SHAP to be added later
        "confidence": 0.8,   # placeholder
        "request_id": rid,
    }

@app.post("/api/v1/batch/score")
def batch_score(req: BatchRequest, x_api_key: str = Header(default="")):
    t0 = time.time()

    # Auth
    if x_api_key != API_KEY:
        _log_req(uuid.uuid4().hex[:8], 0.0, "unauthorized", 0.0, 401, "/api/v1/batch/score", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid API key")

    results = []
    ok = True

    for rec in req.records:
        rid = uuid.uuid4().hex[:8]
        try:
            df = _df_from_payload(rec)
            prob_default = float(PIPE.predict_proba(df)[0, 1])
            threshold = 0.25
            decision = "approve" if prob_default < threshold else "review"

            results.append({
                "request_id": rid,
                "prob_default": round(prob_default, 4),
                "decision": decision,
                "threshold": threshold
            })
            _log_req(rid, 0.0, decision, prob_default, 200, "/api/v1/batch/score", x_api_key)
        except Exception as e:
            ok = False
            results.append({"request_id": rid, "error": str(e)})
            _log_req(rid, 0.0, "error", 0.0, 400, "/api/v1/batch/score", x_api_key)

    total_latency = (time.time() - t0) * 1000.0
    return {
        "model_version": MANIFEST["model_version"],
        "count": len(results),
        "latency_ms": round(total_latency, 2),
        "results": results,
        "status": "ok" if ok else "partial"
    }
