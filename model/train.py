# model/train.py
import json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
ART = Path(__file__).parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Loan_Status"  # Y/N approvals

def ks_stat(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    pos = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    neg = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    pos = pos.fillna(method="ffill").fillna(0)
    neg = neg.fillna(method="ffill").fillna(0)
    return float(np.max(np.abs(pos - neg)))

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Expected CSV at {RAW}. Put Kaggle Loan Prediction train.csv there.")

    df = pd.read_csv(RAW)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in {RAW}")

    y = (df[TARGET_COL].map({"Y": 1, "N": 0})).astype(int)
    X = df.drop(columns=[TARGET_COL, "Loan_ID"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, p_test)
    pr = average_precision_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    ks = ks_stat(y_test.values, p_test)

    feature_cols = X.columns.tolist()
    joblib.dump({"pipeline": pipe, "feature_cols": feature_cols}, ART / "model_v1_0_0.joblib")

    manifest = {
        "model_version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": "scikit-learn",
        "target": "approved",
        "metrics": {
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "brier": round(float(brier), 4),
            "ks": round(float(ks), 4),
        },
        "features": feature_cols,
        "calibration": "none",
        "notes": "Kaggle Loan Prediction; LogisticRegression baseline; seed=42",
    }
    with open(ART / "manifest_v1_0_0.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved:", (ART / "model_v1_0_0.joblib").as_posix())
    print("Manifest:", (ART / "manifest_v1_0_0.json").as_posix())
    print("Metrics:", manifest["metrics"])

if __name__ == "__main__":
    main()
