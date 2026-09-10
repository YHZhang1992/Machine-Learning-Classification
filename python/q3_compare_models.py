from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common import MODEL_ORDER, candidate_feature_counts, load_xy, nested_cv_model, recommend_model, summarize_outer_results, write_json


def parse_k_values(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Q3: compare model families with feature selection and hyperparameter tuning nested inside CV.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--feature-counts", default=None)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/q3_model_comparison")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X, y, labels, _ = load_xy(args.data, args.label, args.id_column, args.positive)
    k_values = candidate_feature_counts(X.shape[1], parse_k_values(args.feature_counts))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Supported models: {MODEL_ORDER}")

    all_folds = []
    all_params = {}
    for model_name in models:
        fold_results, _, best_params = nested_cv_model(X, y, model_name, k_values, args.outer_folds, args.inner_folds, args.seed)
        all_folds.append(fold_results)
        all_params[model_name] = best_params

    fold_table = pd.concat(all_folds, ignore_index=True)
    summary = summarize_outer_results(fold_table).sort_values("mean_roc_auc", ascending=False)
    recommended = recommend_model(summary)
    fold_table.to_csv(out / "outer_fold_metrics.csv", index=False)
    summary.to_csv(out / "model_comparison_summary.csv", index=False)
    (out / "recommended_model.txt").write_text(recommended + "\n", encoding="utf-8")
    write_json(out / "model_comparison_metadata.json", {
        "question": "Q3 - compare model families and select the most suitable model",
        "primary_metric": "ROC AUC",
        "selection_rule": "one-standard-error rule on mean outer-fold ROC AUC, then simplicity preference: " + " > ".join(MODEL_ORDER),
        "positive_label": labels.positive,
        "negative_label": labels.negative,
        "candidate_feature_counts": k_values,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "recommended_model": recommended,
        "outer_best_params": all_params,
    })


if __name__ == "__main__":
    main()
