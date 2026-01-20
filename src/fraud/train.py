from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud.data_loader import load_data
from fraud.features import FeatureConfig, build_features, split_xy


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: dict,
) -> Pipeline:
    clf = LogisticRegression(
        max_iter=int(model_cfg["max_iter"]),
        class_weight=model_cfg.get("class_weight"),
        random_state=int(model_cfg["random_state"]),
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("model", clf),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main(config_path: str = "configs/train.yaml") -> None:
    cfg = load_config(config_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Load data (schema already enforced)
    df_train = load_data(
        cfg["data"]["train_path"],
        cfg["data"]["schema_path"],
    )
    df_valid = load_data(
        cfg["data"]["valid_path"],
        cfg["data"]["schema_path"],
    )

    # Split label and features
    X_train_raw, y_train = split_xy(df_train)
    X_valid_raw, y_valid = split_xy(df_valid)

    feat_cfg = FeatureConfig(
        category_top_k=int(cfg["features"]["category_top_k"]),
        include_merchant=bool(cfg["features"]["include_merchant"]),
    )

    # Build features
    X_train = build_features(X_train_raw, config=feat_cfg)
    X_valid = build_features(X_valid_raw, config=feat_cfg)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_type": cfg["model"]["type"],
                "random_state": cfg["model"]["random_state"],
                "max_iter": cfg["model"]["max_iter"],
                "class_weight": cfg["model"]["class_weight"],
                "category_top_k": cfg["features"]["category_top_k"],
                "include_merchant": cfg["features"]["include_merchant"],
            }
        )

        # Train and evaluate
        model = train_model(X_train, y_train, cfg["model"])
        metrics = evaluate_model(model, X_valid, y_valid)
        mlflow.log_metrics(metrics)

        # 🔑 Infer signature for training–serving parity
        signature = infer_signature(
            X_train,
            model.predict_proba(X_train),
        )

        # 🔑 Log & REGISTER model in MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name="fraud-risk-model",
        )

        # Save feature columns (for serving parity later)
        Path("artifacts").mkdir(exist_ok=True)
        feature_cols_path = "artifacts/feature_columns.json"
        with open(feature_cols_path, "w") as file:
            json.dump(list(X_train.columns), file)
        mlflow.log_artifact(feature_cols_path)

    print("Training complete. Metrics:", metrics)


if __name__ == "__main__":
    main()
