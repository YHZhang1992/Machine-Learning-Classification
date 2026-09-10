from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from common import candidate_feature_counts, load_xy, nested_cv_model, summarize_outer_results, write_json


def parse_k_values(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Q1: optimize feature count and quantify feature stability using leakage-safe nested CV.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--feature-counts", default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/q1_feature_optimization")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X, y, labels, _ = load_xy(args.data, args.label, args.id_column, args.positive)
    k_values = candidate_feature_counts(X.shape[1], parse_k_values(args.feature_counts))

    fold_results, selections, best_params = nested_cv_model(X, y, "ElasticNet", k_values, args.outer_folds, args.inner_folds, args.seed)
    fold_results.to_csv(out / "outer_fold_metrics.csv", index=False)
    summarize_outer_results(fold_results).to_csv(out / "performance_summary.csv", index=False)

    counts = Counter(feature for selected in selections for feature in selected)
    stability = pd.DataFrame({"feature": list(X.columns), "selection_count": [counts.get(f, 0) for f in X.columns]})
    stability["selection_rate"] = stability["selection_count"] / len(selections)
    stability = stability.sort_values(["selection_rate", "feature"], ascending=[False, True])
    stability.to_csv(out / "feature_stability.csv", index=False)

    final_k = int(np.median(fold_results["best_k"]))
    final_k = max(1, min(final_k, X.shape[1]))
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    selector = SelectKBest(score_func=f_classif, k=final_k).fit(X_imp, y)
    final_features = X.columns[selector.get_support()].tolist()
    (out / "candidate_feature_panel.txt").write_text("\n".join(final_features) + "\n", encoding="utf-8")

    write_json(out / "feature_optimization_metadata.json", {
        "question": "Q1 - optimize predictive feature combination",
        "reference_model": "ElasticNet",
        "feature_selector": "SelectKBest(f_classif) fit inside each inner-CV training fold",
        "candidate_feature_counts": k_values,
        "final_candidate_k": final_k,
        "positive_label": labels.positive,
        "negative_label": labels.negative,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "note": "candidate_feature_panel.txt is a full-development-data panel for downstream freezing, not an unbiased CV evaluation set. Q3 reruns selection inside nested CV.",
        "outer_best_params": best_params,
    })


if __name__ == "__main__":
    main()
