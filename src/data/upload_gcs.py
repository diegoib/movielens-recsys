"""Upload raw and processed data to GCS bucket <project_id>-data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def upload(project_id: str) -> None:
    bucket = f"gs://{project_id}-data"

    pairs = [
        (RAW_DIR, f"{bucket}/raw/"),
        (PROCESSED_DIR, f"{bucket}/processed/"),
    ]

    for local_dir, gcs_prefix in pairs:
        if not local_dir.exists():
            print(f"Skipping {local_dir} (not found)")
            continue
        print(f"Uploading {local_dir}/ → {gcs_prefix}")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", str(local_dir) + "/.", gcs_prefix],
            check=True,
        )

    print("Upload complete.")


if __name__ == "__main__":
    pid = os.environ.get("GCP_PROJECT_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not pid:
        sys.exit("Usage: GCP_PROJECT_ID=<id> uv run python src/data/upload_gcs.py")
    upload(pid)
