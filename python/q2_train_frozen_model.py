from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib

from common import MODEL_ORDER, candidate_feature_counts, fit_frozen_model, load_xy, package_versions, write_json


def parse_k_values(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_model_name(path, explicit):
    if explicit:
        return explicit
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --model or --recommended-model-file.")


def main():
    parser = argparse.ArgumentParser(description="Q2: fit and serialize the final transfer-ready frozen classifier after model choice is locked.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--model", default=None, choices=MODEL_ORDER)
    parser.add_argument("--recommended-model-file", default=None)
    parser.add_argument("--feature-counts", default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/python_frozen_model")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X, y, labels, _ = load_xy(args.data, args.label, args.id_column, args.positive)
    k_values = candidate_feature_counts(X.shape[1], parse_k_values(args.feature_counts))
    model_name = read_model_name(args.recommended_model_file, args.model)
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {MODEL_ORDER}")

    final_pipe, selected_features, best_params, inner_auc = fit_frozen_model(X, y, model_name, k_values, args.inner_folds, args.seed)
    artifact = {
        "pipeline": final_pipe,
        "selected_features": selected_features,
        "model_name": model_name,
        "positive_label": labels.positive,
        "negative_label": labels.negative,
        "training_feature_columns": X.columns.tolist(),
        "best_search_params": best_params,
        "inner_cv_roc_auc": inner_auc,
        "seed": args.seed,
        "package_versions": package_versions(),
    }
    joblib.dump(artifact, out / "frozen_model.joblib")
    (out / "selected_features.txt").write_text("\n".join(selected_features) + "\n", encoding="utf-8")
    metadata = {k: v for k, v in artifact.items() if k != "pipeline"}
    metadata.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Q2 - train a frozen model for transfer",
        "important": "Performance should be reported from Q3 nested CV and/or an untouched external validation cohort, not from this full-development-data refit.",
    })
    write_json(out / "frozen_model_metadata.json", metadata)


if __name__ == "__main__":
    main()
