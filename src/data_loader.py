"""
data_loader.py — Load and merge the CMU Movie Summary Corpus datasets.

Files used:
  - movie.metadata.tsv  (81,741 rows)
  - plot_summaries.txt   (42,306 rows)
"""

import json
import pandas as pd


# Column names for movie.metadata.tsv (no header in file)
META_COLUMNS = [
    "wiki_id",
    "freebase_id",
    "name",
    "release_date",
    "revenue",
    "runtime",
    "languages",
    "countries",
    "genres_raw",
]


def load_metadata(path: str) -> pd.DataFrame:
    """Load movie.metadata.tsv into a DataFrame."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=META_COLUMNS,
        dtype={"wiki_id": str},
        quoting=3,  # QUOTE_NONE — handles stray quotes
    )
    return df


def load_plots(path: str) -> pd.DataFrame:
    """Load plot_summaries.txt into a DataFrame with columns [wiki_id, plot]."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                rows.append({"wiki_id": parts[0].strip(), "plot": parts[1].strip()})
    df = pd.DataFrame(rows)
    return df


def _parse_genres(raw: str) -> list:
    """Parse the JSON-encoded genre dict into a list of genre names."""
    try:
        genre_dict = json.loads(raw)
        return list(genre_dict.values())
    except (json.JSONDecodeError, TypeError):
        return []


def merge_data(meta_df: pd.DataFrame, plot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join metadata with plot summaries on wiki_id.
    Adds a 'genres' column (list of genre name strings).
    Drops rows with no genres or no plot.
    """
    merged = pd.merge(meta_df, plot_df, on="wiki_id", how="inner")

    # Parse genre JSON
    merged["genres"] = merged["genres_raw"].apply(_parse_genres)

    # Drop rows with empty genres or missing plot
    merged = merged[merged["genres"].apply(len) > 0].copy()
    merged = merged[merged["plot"].str.strip().astype(bool)].copy()

    merged.reset_index(drop=True, inplace=True)
    return merged


def filter_top_genres(df: pd.DataFrame, top_n: int = 20) -> tuple:
    """
    Keep only the top-N most frequent genres.

    Returns:
        (filtered_df, top_genre_names)
    """
    from collections import Counter

    # Count genre occurrences
    genre_counter = Counter()
    for genre_list in df["genres"]:
        genre_counter.update(genre_list)

    top_genres = [g for g, _ in genre_counter.most_common(top_n)]

    # Filter each movie's genre list to keep only top genres
    df = df.copy()
    df["genres"] = df["genres"].apply(
        lambda gl: [g for g in gl if g in top_genres]
    )

    # Drop movies that have zero genres left
    df = df[df["genres"].apply(len) > 0].copy()
    df.reset_index(drop=True, inplace=True)

    return df, sorted(top_genres)


if __name__ == "__main__":
    # Quick sanity check
    meta = load_metadata("movie.metadata.tsv")
    plots = load_plots("plot_summaries.txt")
    merged = merge_data(meta, plots)
    filtered, genre_names = filter_top_genres(merged)
    print(f"Merged rows:   {len(merged)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Top genres:    {genre_names}")
