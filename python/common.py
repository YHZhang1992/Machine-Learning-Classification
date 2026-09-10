from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_ORDER = ["ElasticNet", "NaiveBayes", "SVM", "RandomForest"]


@dataclass(frozen=True)
class LabelEncoding:
    negative: str
    positive: str


def load_xy(csv_path: str | Path, label_column: str, id_column: str | None = None, positive_label: str | None = None):
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' was not found in {csv_path}.")
    ids = None
    if id_column:
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' was not found in {csv_path}.")
        ids = df[id_column].copy()
    raw_y = df[label_column].astype(str)
    levels = sorted(raw_y.dropna().unique().tolist())
    if len(levels) != 2:
        raise ValueError(f"Binary classification requires exactly 2 outcome levels; found {levels}.")
    positive = positive_label if positive_label is not None else levels[1]
    if positive not in levels:
        raise ValueError(f"Positive label '{positive}' is not one of the observed labels: {levels}.")
    negative = levels[0] if levels[1] == positive else levels[1]
    y = (raw_y.to_numpy() == positive).astype(int)
    drop_cols = [label_column] + ([id_column] if id_column else [])
    X = df.drop(columns=drop_cols)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError("All predictors must be numeric. Non-numeric columns: " + ", ".join(non_numeric[:20]))
    if X.shape[1] < 1:
        raise ValueError("No predictor columns remain after removing the label/ID columns.")
    return X, y, LabelEncoding(negative=negative, positive=positive), ids


def candidate_feature_counts(p: int, requested: Iterable[int] | None = None) -> list[int]:
    defaults = [5, 10, 20, 30, 50, 100, p]
    values = list(requested) if requested is not None else defaults
    values = sorted({int(k) for k in values if int(k) > 0 and int(k) <= p})
    if p not in values:
        values.append(p)
    return sorted(set(values))


def build_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "ElasticNet":
        model = LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced", random_state=seed)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1)
    elif model_name == "NaiveBayes":
        model = GaussianNB()
    elif model_name == "SVM":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def model_grid(model_name: str, k_values: list[int]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {"select__k": k_values}
    if model_name == "ElasticNet":
        grid.update({"model__C": [0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.0, 0.5, 1.0]})
    elif model_name == "RandomForest":
        grid.update({"model__max_features": ["sqrt", 0.5], "model__min_samples_leaf": [1, 5]})
    elif model_name == "NaiveBayes":
        grid.update({"model__var_smoothing": [1e-11, 1e-9, 1e-7]})
    elif model_name == "SVM":
        grid.update({"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01, 0.1]})
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return grid


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def selected_features_from_pipeline(pipe: Pipeline, feature_names: pd.Index) -> list[str]:
    return feature_names[pipe.named_steps["select"].get_support()].tolist()


def nested_cv_model(X: pd.DataFrame, y: np.ndarray, model_name: str, k_values: list[int], outer_splits: int = 5, inner_splits: int = 4, seed: int = 42):
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    fold_rows, selections, best_params = [], [], []
    for fold_id, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1000 + fold_id)
        search = GridSearchCV(build_pipeline(model_name, seed + fold_id), model_grid(model_name, k_values), scoring="roc_auc", cv=inner, refit=True, n_jobs=-1, error_score="raise")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        search.fit(X_train, y_train)
        prob = search.best_estimator_.predict_proba(X_test)[:, 1]
        fold_rows.append({"model": model_name, "outer_fold": fold_id, "inner_best_roc_auc": float(search.best_score_), "best_k": int(search.best_params_["select__k"]), **classification_metrics(y_test, prob)})
        selections.append(selected_features_from_pipeline(search.best_estimator_, X.columns))
        best_params.append(dict(search.best_params_))
    return pd.DataFrame(fold_rows), selections, best_params


def summarize_outer_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier"]
    rows = []
    for model, group in fold_results.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "n_outer_folds": int(group.shape[0])}
        for metric in metrics:
            vals = group[metric].astype(float)
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"se_{metric}"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        row["median_best_k"] = int(np.median(group["best_k"]))
        rows.append(row)
    return pd.DataFrame(rows)


def recommend_model(summary: pd.DataFrame) -> str:
    best_idx = summary["mean_roc_auc"].idxmax()
    best_mean = float(summary.loc[best_idx, "mean_roc_auc"])
    best_se = float(summary.loc[best_idx, "se_roc_auc"])
    eligible = summary.loc[summary["mean_roc_auc"] >= best_mean - best_se, "model"].tolist()
    for model in MODEL_ORDER:
        if model in eligible:
            return model
    return str(summary.loc[best_idx, "model"])


def fit_frozen_model(X: pd.DataFrame, y: np.ndarray, model_name: str, k_values: list[int], inner_splits: int = 5, seed: int = 42):
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(build_pipeline(model_name, seed), model_grid(model_name, k_values), scoring="roc_auc", cv=inner, refit=True, n_jobs=-1, error_score="raise")
    search.fit(X, y)
    selected = selected_features_from_pipeline(search.best_estimator_, X.columns)
    final_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", clone(search.best_estimator_.named_steps["model"])),
    ])
    final_pipe.fit(X[selected], y)
    return final_pipe, selected, dict(search.best_params_), float(search.best_score_)


def package_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
