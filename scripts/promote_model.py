from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--roc-auc-min", type=float, default=0.88)
    p.add_argument("--recall-min", type=float, default=0.65)
    return p.parse_args()


def get_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    run = client.get_run(run_id)
    return run.data.metrics


def main() -> None:
    args = parse_args()

    client = MlflowClient()
    model_name = args.model_name

    # Get latest model version
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))

    metrics = get_run_metrics(client, latest.run_id)

    print(f"Evaluating model v{latest.version} with metrics: {metrics}")

    # ---- STAGING GATE ----
    if metrics.get("roc_auc", 0.0) < args.roc_auc_min:
        raise SystemExit("FAIL: roc_auc below staging threshold")

    if metrics.get("recall", 0.0) < args.recall_min:
        raise SystemExit("FAIL: recall below staging threshold")

    print("PASS: eligible for Staging")

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # ---- PRODUCTION COMPARISON ----
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        print("No Production model found → promoting to Production")
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=False,
        )
        return

    prod = prod_versions[0]
    prod_metrics = get_run_metrics(client, prod.run_id)

    if metrics["roc_auc"] > prod_metrics.get("roc_auc", 0.0):
        print("New model beats Production → promoting")
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=False,
        )
    else:
        print("Model does NOT beat Production → staying in Staging")


if __name__ == "__main__":
    main()
