# api/main.py
import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Header, HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# -------------------------------
# SHAP (optional but preferred)
# -------------------------------
try:
    import shap  # type: ignore
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

with open(MANIFEST_PATH, "r") as f:
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
          top_contrib_json TEXT,
          rule_flags_json TEXT,
          status_code INTEGER NOT NULL,
          endpoint TEXT NOT NULL,
          client_api_key TEXT
        );
    """))

with ENGINE.begin() as conn:
    try:
        conn.execute(text("ALTER TABLE request_logs ADD COLUMN top_contrib_json TEXT"))
    except Exception:
        pass
    try:
        conn.execute(text("ALTER TABLE request_logs ADD COLUMN rule_flags_json TEXT"))
    except Exception:
        pass

def _log_req(request_id: str,
             latency_ms: float,
             decision: str,
             prob_default: float,
             status_code: int,
             endpoint: str,
             api_key_value: str,
             top_contrib_json: str | None,
             rule_flags_json: str | None):
    with ENGINE.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO request_logs
                (ts, request_id, model_version, latency_ms, decision,
                 prob_default, top_contrib_json, rule_flags_json, status_code, endpoint, client_api_key)
                VALUES (:ts, :rid, :ver, :lat, :dec, :prob, :top, :rules, :code, :ep, :api)
            """),
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rid": request_id,
                "ver": MANIFEST["model_version"],
                "lat": float(latency_ms),
                "dec": decision,
                "prob": float(prob_default),
                "top": top_contrib_json,
                "rules": rule_flags_json,
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
        # Permutation explainer works with arbitrary pipelines
        SHAP_EXPLAINER = shap.Explainer(PIPE, BG_DF, algorithm="permutation")
        SHAP_READY = True
    except Exception:
        SHAP_READY = False
        SHAP_EXPLAINER = None

_init_shap()

def _df_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame in the model's expected column order."""
    row = {col: payload.get(col, None) for col in FEATURES}
    return pd.DataFrame([row])

def _linear_explanations(df_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Fallback using LogisticRegression contributions on transformed features."""
    try:
        pre = PIPE.named_steps["pre"]
        clf = PIPE.named_steps["clf"]
    except Exception:
        return []
    Xtr = pre.transform(df_row)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    X = np.asarray(Xtr)[0]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(X))]
    if not hasattr(clf, "coef_"):
        return []
    contrib = X * clf.coef_[0]
    idx = np.argsort(np.abs(contrib))[::-1][:k]
    out = []
    for i in idx:
        fname = str(names[i]).replace("num__", "").replace("cat__", "")
        out.append({"feature": fname, "impact": "higher_approval" if contrib[i] > 0 else "higher_risk"})
    return out

def _top_explanations(df_row: pd.DataFrame, prob_approved: float, k: int = 3) -> list[dict]:
    """Try SHAP; if anything’s off, fall back to linear contributions."""
    if not SHAP_READY or SHAP_EXPLAINER is None:
        return _linear_explanations(df_row, k=k)
    try:
        sv = SHAP_EXPLAINER(df_row)               # 1 x n_features
        vals = np.ravel(np.asarray(sv.values))     # -> (n_features,)
        names = list(sv.feature_names or df_row.columns)
        if len(vals) != len(names):
            names = list(df_row.columns)
        if len(vals) != len(names):
            return _linear_explanations(df_row, k=k)
        s = pd.Series(vals, index=names)
        top = s.abs().sort_values(ascending=False).head(k)
        return [
            {"feature": str(feat),
             "impact": ("higher_approval" if s[feat] > 0 else "higher_risk")}
            for feat in top.index
        ]
    except Exception:
        return _linear_explanations(df_row, k=k)

def compute_linear_contributions(df_row: pd.DataFrame, k: int = 10) -> list[dict]:
    """Return top-k transformed feature contributions for Power BI."""
    try:
        pre = PIPE.named_steps["pre"]
        clf = PIPE.named_steps["clf"]
    except Exception:
        return []
    try:
        Xtr = pre.transform(df_row)
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray()
        X = np.asarray(Xtr)[0]
        names = pre.get_feature_names_out()
        if not hasattr(clf, "coef_"):
            return []
        contrib = X * clf.coef_[0]
        idx = np.argsort(np.abs(contrib))[::-1][:k]
        return [
            {"feature": str(names[i]), "contribution": float(contrib[i])}
            for i in idx
        ]
    except Exception:
        return []

def _rule_flags(payload: Dict[str, Any]) -> list[str]:
    flags: list[str] = []
    applicant_income = _to_float(payload.get("ApplicantIncome"))
    if applicant_income is None or applicant_income <= 0:
        flags.append("income_missing_or_zero")
    loan_amount = _to_float(payload.get("LoanAmount"))
    if loan_amount is None or loan_amount <= 0:
        flags.append("loanamount_missing_or_invalid")
    elif loan_amount > 1000:
        flags.append("out_of_distribution_loanamount")
    return flags

def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _score_payload(payload: Dict[str, Any]) -> dict:
    """Core scorer used by both API and demo."""
    rule_flags = _rule_flags(payload)
    if "income_missing_or_zero" in rule_flags:
        return {
            "model_version": MANIFEST["model_version"],
            "prob_approved": 0.0,
            "prob_default": 1.0,
            "decision": "reject",
            "threshold": 0.25,
            "explanations": [],
            "confidence": 0.8,
            "explanation": "income_missing_or_zero",
            "rule_flags": rule_flags,
            "top_contributions": [],
        }

    if "loanamount_missing_or_invalid" in rule_flags:
        return {
            "model_version": MANIFEST["model_version"],
            "prob_approved": 0.5,
            "prob_default": 0.5,
            "decision": "review",
            "threshold": 0.25,
            "explanations": [],
            "confidence": 0.8,
            "explanation": "loanamount_missing_or_invalid",
            "rule_flags": rule_flags,
            "top_contributions": [],
        }

    force_review = "out_of_distribution_loanamount" in rule_flags
    df = _df_from_payload(payload)
    prob_approved = float(PIPE.predict_proba(df)[0, 1])
    prob_default = 1.0 - prob_approved
    threshold = 0.25

    if prob_default <= 0.25:
        decision = "approve"
    elif prob_default < 0.50:
        decision = "review"
    else:
        decision = "reject"

    explanation = None
    if force_review:
        decision = "review"
        explanation = "out_of_distribution_loanamount"

    explanations = _top_explanations(df, prob_approved, k=3)
    top_contributions = compute_linear_contributions(df, k=10)
    return {
        "model_version": MANIFEST["model_version"],
        "prob_approved": round(prob_approved, 4),
        "prob_default": round(prob_default, 4),
        "decision": decision,
        "threshold": threshold,
        "explanations": explanations,
        "confidence": 0.8,
        "explanation": explanation,
        "rule_flags": rule_flags,
        "top_contributions": top_contributions,
    }

# -------------------------------
# FastAPI app + schemas
# -------------------------------
app = FastAPI(title="LoanLens API", version=APP_VERSION)

class ScoreRequest(BaseModel):
    payload: Dict[str, Any]

class BatchRequest(BaseModel):
    records: list[dict]

# -------------------------------
# Landing + Demo (human-friendly)
# -------------------------------
@app.get("/", include_in_schema=False)
def root_page():
    return HTMLResponse("""
    <h1>LoanRisk API</h1>
    <p style="max-width:720px">
      LoanRisk is a production-style credit risk scoring microservice. 
      Send applicant data and receive a calibrated probability of default, an approve/review decision, 
      and the top factors driving that decision. Try the live demo or integrate via the OpenAPI docs.
    </p>
    <ul>
      <li><a href="/demo">Live Demo</a> – fill a form and get a decision + explanations</li>
      <li><a href="/docs">API Docs</a> – OpenAPI/Swagger (try endpoints in the browser)</li>
      <li><a href="/api/v1/health">Health</a> – model status & version</li>
      <li><a href="https://github.com/vsm34/loanrisk" target="_blank">GitHub Repository</a></li>
    </ul>
    """)


@app.get("/demo", include_in_schema=False)
def demo_page():
    return HTMLResponse("""
<!doctype html><html><head><meta charset="utf-8" />
<title>LoanRisk Demo</title>
<style>
body{font-family:system-ui, sans-serif; max-width:720px; margin:2rem auto; line-height:1.4}
input,select{padding:6px;width:100%} label{font-weight:600;margin-top:.75rem;display:block}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin-top:1rem}
button{padding:10px 14px;border-radius:10px;border:1px solid #222;cursor:pointer}
code{background:#f6f8fa;padding:2px 4px;border-radius:4px}
</style>
</head>
<body>
  <h1>LoanRisk Demo</h1>
  <p>Try a sample applicant and get an approval decision with explanation.</p>
  <div class="card">
    <div class="row">
      <div>
        <label>Gender</label>
        <select id="Gender"><option>Male</option><option>Female</option></select>
      </div>
      <div>
        <label>Married</label>
        <select id="Married"><option>Yes</option><option>No</option></select>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Dependents</label>
        <input id="Dependents" value="0" />
      </div>
      <div>
        <label>Education</label>
        <select id="Education"><option>Graduate</option><option>Not Graduate</option></select>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Self Employed</label>
        <select id="Self_Employed"><option>No</option><option>Yes</option></select>
      </div>
      <div>
        <label>Property Area</label>
        <select id="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Applicant Income</label>
        <input id="ApplicantIncome" type="number" value="5849" />
      </div>
      <div>
        <label>Coapplicant Income</label>
        <input id="CoapplicantIncome" type="number" value="0" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Loan Amount</label>
        <input id="LoanAmount" type="number" value="128" />
      </div>
      <div>
        <label>Loan Term (months)</label>
        <input id="Loan_Amount_Term" type="number" value="360" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Credit History (0 or 1)</label>
        <input id="Credit_History" type="number" value="1" />
      </div>
    </div>

    <p><button id="go">Score Applicant</button></p>
    <pre id="out"></pre>
  </div>

<script>
document.getElementById('go').onclick = async () => {
  const payload = {
    Gender: document.getElementById('Gender').value,
    Married: document.getElementById('Married').value,
    Dependents: document.getElementById('Dependents').value,
    Education: document.getElementById('Education').value,
    Self_Employed: document.getElementById('Self_Employed').value,
    ApplicantIncome: Number(document.getElementById('ApplicantIncome').value),
    CoapplicantIncome: Number(document.getElementById('CoapplicantIncome').value),
    LoanAmount: Number(document.getElementById('LoanAmount').value),
    Loan_Amount_Term: Number(document.getElementById('Loan_Amount_Term').value),
    Credit_History: Number(document.getElementById('Credit_History').value),
    Property_Area: document.getElementById('Property_Area').value
  };
  const res = await fetch('/demo/score', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
};
</script>
</body></html>
    """)

@app.post("/demo/score", include_in_schema=False)
def demo_score(payload: Dict[str, Any]):
    """
    Public demo scorer: no API key required.
    Calls the same model server-side (no key exposed in browser).
    """
    try:
        result = _score_payload(payload)
        return JSONResponse({"ok": True, "result": result})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

# -------------------------------
# API endpoints (machine-facing)
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

class ScoreRequest(BaseModel):
    payload: Dict[str, Any]

class BatchRequest(BaseModel):
    records: list[dict]

@app.post("/api/v1/score")
def score(req: ScoreRequest, x_api_key: str = Header(default="")):
    t0 = time.time()
    if x_api_key != API_KEY:
        _log_req(uuid.uuid4().hex[:8], 0.0, "unauthorized", 0.0, 401, "/api/v1/score", x_api_key, None, None)
        raise HTTPException(status_code=401, detail="Invalid API key")

    rid = uuid.uuid4().hex[:8]
    try:
        result = _score_payload(req.payload)
        decision = result["decision"]
        prob_default = result["prob_default"]
    except Exception as e:
        _log_req(rid, (time.time() - t0) * 1000.0, "error", 0.0, 400, "/api/v1/score", x_api_key, None, None)
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")

    latency_ms = (time.time() - t0) * 1000.0
    rule_flags = result.get("rule_flags", [])
    top_contrib_json = None
    if "income_missing_or_zero" not in rule_flags and "loanamount_missing_or_invalid" not in rule_flags:
        df_row = _df_from_payload(req.payload)
        top_contrib_json = json.dumps(compute_linear_contributions(df_row, k=10))
    rule_flags_json = json.dumps(rule_flags)
    _log_req(rid, latency_ms, decision, prob_default, 200, "/api/v1/score", x_api_key, top_contrib_json, rule_flags_json)

    result["request_id"] = rid
    return result

@app.post("/api/v1/batch/score")
def batch_score(req: BatchRequest, x_api_key: str = Header(default="")):
    t0 = time.time()
    if x_api_key != API_KEY:
        _log_req(uuid.uuid4().hex[:8], 0.0, "unauthorized", 0.0, 401, "/api/v1/batch/score", x_api_key, None, None)
        raise HTTPException(status_code=401, detail="Invalid API key")

    results = []
    ok = True
    for rec in req.records:
        rid = uuid.uuid4().hex[:8]
        try:
            r = _score_payload(rec)
            r["request_id"] = rid
            results.append(r)
            rule_flags = r.get("rule_flags", [])
            top_contrib_json = None
            if "income_missing_or_zero" not in rule_flags and "loanamount_missing_or_invalid" not in rule_flags:
                df_row = _df_from_payload(rec)
                top_contrib_json = json.dumps(compute_linear_contributions(df_row, k=10))
            rule_flags_json = json.dumps(rule_flags)
            _log_req(rid, 0.0, r["decision"], r["prob_default"], 200, "/api/v1/batch/score", x_api_key, top_contrib_json, rule_flags_json)
        except Exception as e:
            ok = False
            results.append({"request_id": rid, "error": str(e)})
            _log_req(rid, 0.0, "error", 0.0, 400, "/api/v1/batch/score", x_api_key, None, None)

    total_latency = (time.time() - t0) * 1000.0
    return {
        "model_version": MANIFEST["model_version"],
        "count": len(results),
        "latency_ms": round(total_latency, 2),
        "results": results,
        "status": "ok" if ok else "partial"
    }

