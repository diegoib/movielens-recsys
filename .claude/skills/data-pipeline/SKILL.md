---
name: data-pipeline
description: Download MovieLens 20M from Kaggle and generate the synthetic events table. Run once before training.
disable-model-invocation: true
---

Run the full offline data pipeline for movielens-recsys.

Prerequisites:
- `uv sync --group data` (includes polars, pandas, pyarrow, kaggle, pydantic, tqdm)
- Env vars set: `KAGGLE_USERNAME` and `KAGGLE_KEY`

Steps:
1. Download MovieLens 20M from Kaggle:
   `uv run python src/data/download.py $ARGUMENTS`
   Output: `data/raw/` (ratings.csv, movies.csv, genome-scores.csv, genome-tags.csv, tags.csv, links.csv)

2. Generate synthetic events table:
   `uv run python src/data/generate_events.py $ARGUMENTS`
   Output: `data/processed/events.parquet` (~150-170M rows)

Event generation logic (@docs/project_summary.md section 2):
- Group ratings into sessions (gap > 60 min = new session)
- Per rated movie: generate impression → view → click → rating funnel (backward from rating timestamp)
- Add negative samples: 4-6 impressions without view per rated movie (40% popular, 40% same-genre, 20% collaborative)
- Add 1 view-without-click per 2-3 clicks (hard negatives)
- Label: click=1, impression/view-without-click=0

Verify output:
- Check row count is in the 150-170M range
- Verify all event_types are present: impression, view, click, rating
- Confirm no user has a negative sample that matches one of their rated movies
