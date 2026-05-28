"""Download MovieLens dataset from grouplens.org."""

import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "ml-latest.zip"

_EXPECTED = {
    "ratings.csv",
    "movies.csv",
    "genome-scores.csv",
    "genome-tags.csv",
    "tags.csv",
    "links.csv",
}


def _download_with_progress(url: str, dest: Path) -> None:
    pbar: tqdm | None = None

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name)
        pbar.update(block_size)

    urllib.request.urlretrieve(url, dest, reporthook)
    if pbar is not None:
        pbar.close()


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading MovieLens from {URL} ...")
        _download_with_progress(URL, ZIP_PATH)
    else:
        print(f"{ZIP_PATH} already exists, skipping download.")

    if all((RAW_DIR / f).exists() for f in _EXPECTED):
        print("All CSV files already present, skipping extraction.")
        return

    print("Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            filename = Path(member.filename).name
            if filename not in _EXPECTED:
                continue
            dest = RAW_DIR / filename
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  {filename}")

    print("Extraction complete.")


if __name__ == "__main__":
    download()
