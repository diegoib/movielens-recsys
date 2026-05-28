# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end recommender system trained on MovieLens 20M with a production stack (RedPanda, PyFlink, Redis, ONNX, MLflow, FastAPI, Prometheus, Grafana). Educational project — see @docs/project_summary.md for full design, data schema, model architecture, and implementation phases.

## Environment

Python 3.12 managed with `uv`. Dependency groups: `data`, `onnx`, `torch`, `train`, `dev`, `test`, `all`.

```bash
uv sync --group all       # install everything
uv sync --group data      # data processing only
uv sync --group train     # training (Lightning, MLflow, sklearn)
```

Run scripts with `uv run python script.py` or `uv run pytest`.

## Code Quality

```bash
uv run ruff format .        # format
uv run ruff check --fix .   # lint + auto-fix
uv run mypy .               # type check (ignore_missing_imports = true)
uv run pytest               # tests
```

## Commits & Branches

Uses **semantic-release** — commits must follow Conventional Commits:
- `feat:` bumps minor, `fix:` bumps patch
- `chore:`, `docs:`, `refactor:`, `test:` do not bump version

Branch naming: `feature/<name>`, `fix/<name>`. All branches except `main` release as `rc` prereleases.

## Data & Artifacts

- Raw data: `data/raw/` (MovieLens CSVs, downloaded from grouplens.org)
- Processed: `data/processed/` (events table ~150-170M rows in Parquet)
- Models: `artifacts/models/`

## Implementation Phases

See @docs/project_summary.md section 4 for the full sequence. In short:
1. Download MovieLens 20M → Simulator 1 (historical events) → Feature engineering
2. Train two-tower DNN → export ONNX → register in MLflow
3. Deploy RedPanda + PyFlink + Redis → FastAPI → Prometheus + Grafana
4. Simulator 2 (real-time events against live infrastructure)

Never use random train/test split — always split **temporally** (train on events before day X, test on day X+1 onward).
