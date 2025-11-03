# api/main.py
import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# SHAP is installed; still wrap safely
try:
    import shap
    HAVE_SHAP = True
except Exception:
    shap = None
    HAVE_SHAP = False

APP_VERSION = "1.0.0"
API_KEY = os.getenv("API_KEY", "devkey")

# -------------------------------
# Model + manifest paths
# -------------------------------
ART_DIR = Path(__file__).parent.parent / "model" / "artifacts"
MODEL_PATH = ART_DIR / "model_v1_0_0.joblib"
MANIFEST_PATH = ART_DIR / "manifest_v1_0_0.json"
BG_PATH = ART_DIR / "background_v1_0_0.csv"

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
                "api": api_key_value[:4] + "****" if api_key_value else None
            }
        )

# -------------------------------
# SHAP setup (Permutation explainer on CSV background)
# -------------------------------
EXPLAIN = True and HAVE_SHAP
SHAP_READY = False
SHAP_EXPLAINER = None
BG_DF = None

def _init_shap():
    global SHAP_READY, SHAP_EXPLAINER, BG_DF
    if not EXPLAIN:
        return
    try:
        if not BG_PATH.exists():
            return
        BG_DF = pd.read_csv(BG_PATH)
        SHAP_EXPLAINER = shap.Explainer(PIPE, BG_DF, algorithm="permutation")
        SHAP_READY = True
    except Exception:
        SHAP_READY = False
        SHAP_EXPLAINER = None

_init_shap()

def _top_explanations(df_row: pd.DataFrame, prob_default: float, k: int = 3) -> list[dict]:
    """
    Try SHAP (permutation) and gracefully fall back to linear contributions.
    Handles both (n_features,) and (1, n_features) shapes from SHAP.
    """
    if not SHAP_READY or SHAP_EXPLAINER is None:
        return _linear_explanations(df_row, k=k)

    try:
        sv = SHAP_EXPLAINER(df_row)  # explanations for 1 row
        vals = np.ravel(np.asarray(sv.values))          # shape -> (n_features,)
        names = list(sv.feature_names or df_row.columns)
        if len(vals) != len(names):
            # mismatch? try just df_row columns, else bail to linear
            names = list(df_row.columns)
        if len(vals) != len(names):
            return _linear_explanations(df_row, k=k)

        s = pd.Series(vals, index=names)
        top = s.abs().sort_values(ascending=False).head(k)
        return [{"feature": str(feat), "impact": ("risk_reducing" if s[feat] < 0 else "high_risk")}
                for feat in top.index]
    except Exception:
        return _linear_explanations(df_row, k=k)


# -------------------------------
# FastAPI app + schemas
# -------------------------------
app = FastAPI(title="LoanLens API", version=APP_VERSION)

class ScoreRequest(BaseModel):
    payload: Dict[str, Any]

class BatchRequest(BaseModel):
    records: list[dict]

def _df_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {col: payload.get(col, None) for col in FEATURES}
    return pd.DataFrame([row])

def _linear_explanations(df_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Fallback: use LogisticRegression coefficients on the transformed row."""
    try:
        pre = PIPE.named_steps["pre"]
        clf = PIPE.named_steps["clf"]
    except Exception:
        return []
    Xtr = pre.transform(df_row)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    Xtr = np.asarray(Xtr)[0]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(Xtr))]
    if not hasattr(clf, "coef_"):
        return []
    contrib = Xtr * clf.coef_[0]
    idx = np.argsort(np.abs(contrib))[::-1][:k]
    out = []
    for i in idx:
        fname = str(names[i]).replace("num__", "").replace("cat__", "")
        out.append({"feature": fname, "impact": "risk_reducing" if contrib[i] > 0 else "high_risk"})
    return out


@app.get("/", include_in_schema=False)
def root_page():
    return HTMLResponse("""
    <h1>LoanLens API</h1>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/api/v1/health">Health</a></li>
      <li>POST <code>/api/v1/score</code>, <code>/api/v1/batch/score</code></li>
    </ul>
    """)

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_version": MANIFEST["model_version"],
        "explainability": bool(SHAP_READY)
    }

@app.get("/api/v1/models/active")
def active_model():
    return MANIFEST

@app.post("/api/v1/score")
def score(req: ScoreRequest, x_api_key: str = Header(default="")):
    t0 = time.time()
    if x_api_key != API_KEY:
        _log_req(uuid.uuid4().hex[:8], 0.0, "unauthorized", 0.0, 401, "/api/v1/score", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid API key")

    df = _df_from_payload(req.payload)
    rid = uuid.uuid4().hex[:8]
    try:
        prob_default = float(PIPE.predict_proba(df)[0, 1])
    except Exception as e:
        _log_req(rid, (time.time() - t0) * 1000.0, "error", 0.0, 400, "/api/v1/score", x_api_key)
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")

    threshold = 0.25
    decision = "approve" if prob_default < threshold else "review"
    latency_ms = (time.time() - t0) * 1000.0
    _log_req(rid, latency_ms, decision, prob_default, 200, "/api/v1/score", x_api_key)

    explanations = _top_explanations(df, prob_default, k=3)

    return {
        "model_version": MANIFEST["model_version"],
        "prob_default": round(prob_default, 4),
        "decision": decision,
        "threshold": threshold,
        "explanations": explanations,
        "confidence": 0.8,
        "request_id": rid,
    }

@app.post("/api/v1/batch/score")
def batch_score(req: BatchRequest, x_api_key: str = Header(default="")):
    t0 = time.time()
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
            explanations = _top_explanations(df, prob_default, k=3)

            results.append({
                "request_id": rid,
                "prob_default": round(prob_default, 4),
                "decision": decision,
                "threshold": threshold,
                "explanations": explanations
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
