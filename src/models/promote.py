"""Promote the latest Staging model to Production if its val_auc improves."""

from __future__ import annotations

import os

import mlflow
import tyro


def promote(
    tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    model_name: str = "two-tower-recsys",
) -> None:
    client = mlflow.MlflowClient(tracking_uri)

    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        raise SystemExit(f"No model in Staging for '{model_name}'")

    production = client.get_latest_versions(model_name, stages=["Production"])

    staging_auc = float(client.get_run(staging[0].run_id).data.metrics["val_auc"])
    prod_auc = (
        float(client.get_run(production[0].run_id).data.metrics["val_auc"]) if production else 0.0
    )

    if staging_auc > prod_auc:
        client.transition_model_version_stage(model_name, staging[0].version, "Production")
        print(f"Promoted {model_name} v{staging[0].version}: {prod_auc:.4f} → {staging_auc:.4f}")
    else:
        print(f"No promotion: Staging val_auc {staging_auc:.4f} ≤ Production {prod_auc:.4f}")


if __name__ == "__main__":
    tyro.cli(promote)
