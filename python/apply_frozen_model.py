from __future__ import annotations

import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Apply a frozen Python model to a transfer/validation cohort.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--output", default="transfer_predictions.csv")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    df = pd.read_csv(args.data)
    selected = artifact["selected_features"]
    missing = [f for f in selected if f not in df.columns]
    if missing:
        raise ValueError(f"Transfer data are missing {len(missing)} required features: " + ", ".join(missing[:20]))

    prob = artifact["pipeline"].predict_proba(df[selected])[:, 1]
    pred01 = (prob >= 0.5).astype(int)
    pred_label = [artifact["positive_label"] if p == 1 else artifact["negative_label"] for p in pred01]
    out = pd.DataFrame({"predicted_probability": prob, "predicted_class": pred_label})
    if args.id_column:
        if args.id_column not in df.columns:
            raise ValueError(f"ID column '{args.id_column}' is absent from transfer data.")
        out.insert(0, args.id_column, df[args.id_column].values)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
