from __future__ import annotations

import argparse
from dataclasses import dataclass

from fraud.data_loader import load_data
from fraud.features import FeatureConfig, build_features, split_xy
from fraud.train import evaluate_model, train_model


@dataclass(frozen=True)
class Gates:
    roc_auc_min: float
    recall_min: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--valid-path", required=True)
    p.add_argument("--schema-path", required=True)
    p.add_argument("--train-n", type=int, default=50_000)
    p.add_argument("--valid-n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)

    # Feature config
    p.add_argument("--category-top-k", type=int, default=50)
    p.add_argument("--include-merchant", action="store_true")

    # Model config
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--class-weight", default="balanced")
    p.add_argument("--random-state", type=int, default=42)

    # Gates
    p.add_argument("--roc-auc-min", type=float, default=0.85)
    p.add_argument("--recall-min", type=float, default=0.60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gates = Gates(
        roc_auc_min=args.roc_auc_min,
        recall_min=args.recall_min,
    )

    df_train = load_data(args.train_path, args.schema_path)
    df_valid = load_data(args.valid_path, args.schema_path)

    # Deterministic sampling to keep CI fast and stable
    df_train = df_train.sample(
        n=min(args.train_n, len(df_train)),
        random_state=args.seed,
    )
    df_valid = df_valid.sample(
        n=min(args.valid_n, len(df_valid)),
        random_state=args.seed,
    )

    X_train_raw, y_train = split_xy(df_train)
    X_valid_raw, y_valid = split_xy(df_valid)

    feat_cfg = FeatureConfig(
        category_top_k=args.category_top_k,
        include_merchant=args.include_merchant,
    )

    X_train = build_features(X_train_raw, config=feat_cfg)
    X_valid = build_features(X_valid_raw, config=feat_cfg)

    # 🔒 CRITICAL: feature alignment
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    model_cfg = {
        "max_iter": args.max_iter,
        "class_weight": (args.class_weight if args.class_weight != "none" else None),
        "random_state": args.random_state,
        "type": "logreg",
    }

    model = train_model(X_train, y_train, model_cfg=model_cfg)
    metrics = evaluate_model(model, X_valid, y_valid)

    print("CI metrics:", metrics)

    if metrics["roc_auc"] < gates.roc_auc_min:
        raise SystemExit(
            f"FAIL: roc_auc {metrics['roc_auc']:.4f} " f"< {gates.roc_auc_min:.4f}"
        )

    if metrics["recall"] < gates.recall_min:
        raise SystemExit(
            f"FAIL: recall {metrics['recall']:.4f} " f"< {gates.recall_min:.4f}"
        )

    print(
        f"PASS: roc_auc {metrics['roc_auc']:.4f} "
        f"(>= {gates.roc_auc_min:.4f}), "
        f"recall {metrics['recall']:.4f} "
        f"(>= {gates.recall_min:.4f})"
    )


if __name__ == "__main__":
    main()
