# scripts/export_logs.py
import sqlite3
from pathlib import Path
import pandas as pd
import joblib

def repo_root() -> Path:
    # scripts/ is one level below root
    return Path(__file__).resolve().parents[1]

def main():
    root = repo_root()

    db_path = root / "loanlens.db"  # change if your db name differs
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found at: {db_path}")

    out_dir = root / "data" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loanrisk_logs.csv"
    weights_path = out_dir / "model_weights.csv"
    intercept_path = out_dir / "model_intercept.csv"

    con = sqlite3.connect(db_path)
    try:
        # quick sanity check: list tables (helps debug table-name issues)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)
        print("Tables in DB:", tables["name"].tolist())

        df = pd.read_sql("SELECT * FROM request_logs;", con)
    finally:
        con.close()

    df.to_csv(out_path, index=False)
    print(f"Exported {len(df):,} rows -> {out_path}")

    model_path = root / "model" / "artifacts" / "model_v1_0_0.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    feature_names = pre.get_feature_names_out()
    weights = clf.coef_[0]
    weights_df = pd.DataFrame({
        "feature": feature_names,
        "weight": weights,
    })
    weights_df["abs_weight"] = weights_df["weight"].abs()
    weights_df = weights_df.sort_values("abs_weight", ascending=False).drop(columns=["abs_weight"])
    weights_df.to_csv(weights_path, index=False)
    print(f"Exported {len(weights_df):,} feature weights -> {weights_path}")

    intercept_df = pd.DataFrame({"intercept": [float(clf.intercept_[0])]})
    intercept_df.to_csv(intercept_path, index=False)
    print(f"Exported model intercept -> {intercept_path}")

if __name__ == "__main__":
    main()
