---
name: train
description: Launch the training pipeline (PyTorch Lightning + MLflow tracking). Use when ready to train or retrain the two-tower recommendation model.
disable-model-invocation: true
---

Run the training pipeline for the movielens-recsys two-tower model.

Prerequisites:
- `uv sync --group train` (includes Lightning, MLflow, sklearn, torch)
- Processed events in `data/processed/` (run `/data-pipeline` first if not there)
- MLflow tracking URI configured (default: local `mlruns/`)

Steps:
1. Check that `data/processed/` contains the events Parquet file
2. Run: `uv run python src/train.py $ARGUMENTS`
3. Monitor training via MLflow: `uv run mlflow ui`
4. After training completes, verify ONNX export in `artifacts/models/`
5. Confirm the model is registered in MLflow with the correct version tag

Architecture reminder (@docs/project_summary.md section 3):
- Two-tower: user tower (user_id embedding + behavior features) + movie tower (movie_id + genre + year + popularity)
- Score = dot product of both towers
- Loss = binary cross-entropy (click=1, no-click=0)
- Evaluation: AUC-ROC (primary), NDCG@5, Precision@5 — always with temporal split
