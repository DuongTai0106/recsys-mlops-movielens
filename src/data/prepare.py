import zipfile
from pathlib import Path

import pandas as pd

from src.common.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    # MovieLens extracts to a folder like ml-20m/
    subfolders = [p for p in extract_to.iterdir() if p.is_dir()]
    if not subfolders:
        raise RuntimeError("Zip extracted but no folder found.")
    return subfolders[0]


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_RAW_DIR / "ml-20m.zip"
    extracted_root = DATA_RAW_DIR / "extracted"
    dataset_dir = extract_zip(zip_path, extracted_root)

    ratings_path = dataset_dir / "ratings.csv"
    movies_path = dataset_dir / "movies.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # ratings: userId, movieId, rating, timestamp
    # Convert to implicit interactions: rating >= 4.0 considered positive
    interactions = ratings.loc[ratings["rating"] >= 4.0, ["userId", "movieId", "timestamp"]].copy()
    interactions.rename(
        columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"}, inplace=True
    )

    # Sort by time (important for later time-based split)
    interactions.sort_values(["user_id", "ts"], inplace=True)

    # Save processed data
    interactions.to_parquet(DATA_PROCESSED_DIR / "interactions.parquet", index=False)
    movies.rename(
        columns={"movieId": "item_id", "title": "title", "genres": "genres"}, inplace=True
    )
    movies.to_parquet(DATA_PROCESSED_DIR / "items.parquet", index=False)

    print("✅ Saved:")
    print(f"- {DATA_PROCESSED_DIR / 'interactions.parquet'}  rows={len(interactions):,}")
    print(f"- {DATA_PROCESSED_DIR / 'items.parquet'}        rows={len(movies):,}")


if __name__ == "__main__":
    main()
