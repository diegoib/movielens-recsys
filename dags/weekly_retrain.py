"""weekly_retrain — Airflow DAG that rebuilds the training dataset and retrains the model.

Schedule: @weekly (every Sunday at 00:00 UTC)

Tasks:
  build_retrain_dataset → train_model → promote_model

Each task is a BashOperator so no Airflow Google Cloud provider is required —
the Airflow VM already has gcloud and uv installed via the Terraform startup script.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

GCS_BUCKET = os.environ.get("GCS_BUCKET", "movielens-recsys-proj-data")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GCP_PROJECT = os.environ.get("GCP_PROJECT_ID", "movielens-recsys-proj")
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")

_env = {
    "GCS_BUCKET": GCS_BUCKET,
    "GCP_REGION": GCP_REGION,
    "GCP_PROJECT": GCP_PROJECT,
}

with DAG(
    dag_id="weekly_retrain",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": 300},
    tags=["recsys", "training"],
) as dag:
    build_dataset = BashOperator(
        task_id="build_retrain_dataset",
        env=_env,
        bash_command=f"""
cd {WORKSPACE}
uv run python src/data/build_retrain_dataset.py \\
    --gcs_inference_path gs://$GCS_BUCKET/inference-logs \\
    --gcs_events_path gs://$GCS_BUCKET/events \\
    --gcs_movies_path gs://$GCS_BUCKET/processed/movie_features.parquet \\
    --output_path gs://$GCS_BUCKET/processed/retrain_{{{{ ds_nodash }}}}.parquet \\
    --since_date 2023-01-01
""",
    )

    train_model = BashOperator(
        task_id="train_model",
        env=_env,
        bash_command="""
gcloud run jobs execute training-job \\
    --region $GCP_REGION \\
    --project $GCP_PROJECT \\
    --args="--gcs_data_path,gs://$GCS_BUCKET/processed/retrain_{{ ds_nodash }}.parquet" \\
    --wait
""",
    )

    promote_model = BashOperator(
        task_id="promote_model",
        env=_env,
        bash_command=f"cd {WORKSPACE} && uv run python src/models/promote.py",
    )

    build_dataset >> train_model >> promote_model
