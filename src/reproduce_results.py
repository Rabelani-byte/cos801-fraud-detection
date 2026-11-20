"""Recompute the main test table from saved test predictions (not hard-coded scores).

Input CSV must contain: y_true, LightGBM, TabNet, FT-Transformer.
All model probabilities must correspond to the SAME chronologically held-out test rows.
Example:
    python src/reproduce_results.py --predictions results/test_predictions.csv

The supplied notebooks currently do not export a unified prediction CSV.
See README for the required additional export step. This script does not retrain models.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, brier_score_loss
)

MODELS = ["LightGBM", "TabNet", "FT-Transformer"]

def ece(y, p, bins=10):
    # Equal-width bins. Must match the notebook's ECE binning to match ECE scores.
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i+1] if i < bins-1 else p <= edges[i+1])
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default="results/recomputed_model_comparison.csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    required = ["y_true", *MODELS]
    missing = [x for x in required if x not in df.columns]
    if missing:
        parser.error(f"Missing columns: {missing}")
    y = df["y_true"].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        parser.error("y_true must contain both binary classes 0 and 1")
    result = []
    for name in MODELS:
        p = df[name].to_numpy(dtype=float)
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            parser.error(f"{name} probabilities must be finite and in [0,1]")
        pred = (p >= args.threshold).astype(int)
        result.append({
            "Model": name,
            "PR-AUC": average_precision_score(y,p),
            "ROC-AUC": roc_auc_score(y,p),
            "Precision": precision_score(y,pred,zero_division=0),
            "Recall": recall_score(y,pred,zero_division=0),
            "F1-Score": f1_score(y,pred,zero_division=0),
            "Brier Score": brier_score_loss(y,p),
            "ECE": ece(y,p)
        })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result).to_csv(args.output,index=False)
    print(pd.DataFrame(result).round(4).to_string(index=False))
    print(f"Saved: {args.output}")
if __name__ == "__main__":
    main()
