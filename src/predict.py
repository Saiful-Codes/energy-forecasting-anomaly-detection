import argparse
import os
import numpy as np
import pandas as pd

from feature_engineering import make_features, TARGET_COL
from model import load_model, load_threshold_config

def parse_args():
    parser = argparse.ArgumentParser(description="Score energy data with forecasting + anomaly flags.")
    parser.add_argument("--input", required=True, help="Path to input CSV (must include datetime + target col).")
    parser.add_argument("--output", required=True, help="Path to output scored CSV.")
    parser.add_argument("--model_path", default="models/forecast_pipeline_v1.joblib", help="Path to saved model.")
    parser.add_argument("--threshold_path", default="models/threshold_v1.json", help="Path to threshold config JSON.")
    parser.add_argument("--datetime_col", default="datetime", help="Name of datetime column in input CSV.")
    parser.add_argument("--target_col", default=TARGET_COL, help="Name of target column in input CSV.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load input
    df = pd.read_csv(args.input, parse_dates=[args.datetime_col])
    df = df.set_index(args.datetime_col).sort_index()

    if args.target_col not in df.columns:
        raise ValueError(f"Input is missing required target column: {args.target_col}")

    # 2) Build features
    X = make_features(df, target_col=args.target_col)

    # Align actuals to feature index (because make_features drops initial rows)
    y_actual = df.loc[X.index, args.target_col]

    # 3) Load model
    model = load_model(args.model_path)

    # 4) Predict
    y_pred = model.predict(X)

    # 5) Residual + abs residual
    residual = y_actual.values - y_pred
    abs_residual = np.abs(residual)

    scored = pd.DataFrame({
        "datetime": X.index,
        "actual": y_actual.values,
        "predicted": y_pred,
        "residual": residual,
        "abs_residual": abs_residual
    }).set_index("datetime")

    # 6) Load threshold config and flag anomalies
    cfg = load_threshold_config(args.threshold_path)

    if cfg.strategy == "p95":
        thr = float(cfg.params["thr_p95"])
        scored["is_anomaly"] = scored["abs_residual"] > thr
        scored["threshold_used"] = thr
    else:
        raise ValueError(f"Unsupported strategy in threshold config: {cfg.strategy}")

    # 7) Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scored.to_csv(args.output)

    print(f"Saved scored output to: {args.output}")
    print(f"Anomalies flagged: {int(scored['is_anomaly'].sum())} / {len(scored)}")

if __name__ == "__main__":
    main()
