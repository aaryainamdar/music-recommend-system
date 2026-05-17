"""
music_engine.py
---------------
Music Mood Analyzer — Core ML Backend

Uses real Spotify track data from five CSV files:
  - data.csv             170 k individual tracks (primary song dataset)
  - data_w_genres.csv    Artist-level data with genre tags
  - data_by_genres.csv   Genre-level aggregates (EDA / summaries)
  - data_by_year.csv     Year-level aggregates  (EDA / trend analysis)
  - data_by_artist.csv   Artist-level aggregates

Audio features (Spotify API spec):
  - acousticness      [0, 1]
  - danceability      [0, 1]
  - energy            [0, 1]
  - instrumentalness  [0, 1]
  - liveness          [0, 1]
  - loudness          [-60, 0] dB
  - speechiness       [0, 1]
  - tempo             BPM
  - valence           [0, 1]
  - popularity        [0, 100]
"""

import ast
import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

PATHS = {
    "tracks":     os.path.join(DATA_DIR, "data.csv"),
    "w_genres":   os.path.join(DATA_DIR, "data_w_genres.csv"),
    "by_genres":  os.path.join(DATA_DIR, "data_by_genres.csv"),
    "by_year":    os.path.join(DATA_DIR, "data_by_year.csv"),
    "by_artist":  os.path.join(DATA_DIR, "data_by_artist.csv"),
}


AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]

DECADE_NAMES = [
    "1920s", "1930s", "1940s", "1950s", "1960s",
    "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
]

CLUSTER_MOOD_LABELS = {
    0: "🎉 Party",
    1: "😌 Chill",
    2: "😢 Melancholic",
    3: "💪 Workout",
    4: "🎻 Acoustic",
    5: "🤖 Electronic",
    6: "🌙 Late Night",
    7: "☀️ Feel-Good",
}

MOOD_PROFILES = {
    "happy": {
        "valence":      (0.80, 2.0),
        "energy":       (0.70, 1.5),
        "danceability": (0.72, 1.2),
    },
    "sad": {
        "valence":      (0.20, 2.0),
        "energy":       (0.30, 1.5),
        "acousticness": (0.60, 1.0),
    },
    "energetic": {
        "energy":       (0.90, 2.5),
        "danceability": (0.85, 1.5),
        "tempo":        (0.75, 1.0),
    },
    "calm": {
        "acousticness": (0.75, 2.0),
        "energy":       (0.25, 2.0),
        "liveness":     (0.10, 1.0),
    },
    "angry": {
        "energy":       (0.90, 2.5),
        "loudness":     (0.85, 1.5),
        "valence":      (0.25, 1.0),
    },
    "romantic": {
        "valence":      (0.65, 2.0),
        "acousticness": (0.55, 1.5),
        "danceability": (0.58, 1.0),
    },
    "focus": {
        "instrumentalness": (0.70, 2.5),
        "energy":           (0.45, 1.5),
        "speechiness":      (0.04, 1.0),
    },
    "party": {
        "danceability": (0.88, 2.5),
        "energy":       (0.85, 1.5),
        "valence":      (0.75, 1.0),
    },
}


def _parse_list_str(value) -> list[str]:
    """
    Safely parse a stringified Python list such as "['rock', 'pop']"
    into an actual Python list.  Returns [] on failure.
    """
    if not isinstance(value, str):
        return []
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _year_to_decade(year: int) -> str:
    """Convert a 4-digit year to a decade string like '1990s'."""
    return f"{(int(year) // 10) * 10}s"


def _clean_artists(value) -> str:
    """
    Artists are stored as stringified lists: "['Radiohead']".
    Return a readable comma-joined string.
    """
    parsed = _parse_list_str(value)
    if parsed:
        return ", ".join(parsed)
    return str(value).strip("[]'\"")


def load_tracks(
    path: str | None = None,
    min_popularity: int = 0,
    year_range: tuple[int, int] = (1920, 2021),
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and clean the main track-level dataset (data.csv).

    Columns retained:
        name, artists, year, decade, release_date, popularity,
        id, explicit, duration_ms,
        + all 9 AUDIO_FEATURES

    Args:
        path           : Override default path from PATHS["tracks"].
        min_popularity : Drop tracks below this popularity score (0–100).
        year_range     : (min_year, max_year) inclusive filter.
        sample_n       : If set, randomly sample this many rows after
                         filtering (useful for fast iteration).
        random_state   : Random seed for sampling.

    Returns:
        pd.DataFrame
    """
    path = path or PATHS["tracks"]
    df   = pd.read_csv(path)

    df["artists"] = df["artists"].apply(_clean_artists)

    df["decade"] = df["year"].apply(_year_to_decade)

    df = df[df["popularity"] >= min_popularity]
    df = df[df["year"].between(*year_range)]

    df = df.dropna(subset=AUDIO_FEATURES)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)

    df = df.reset_index(drop=True)
    print(f"[load_tracks] {len(df):,} tracks loaded.")
    return df


def load_tracks_with_genres(
    tracks_path: str | None = None,
    genres_path: str | None = None,
    min_popularity: int = 0,
    year_range: tuple[int, int] = (1920, 2021),
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load track-level data and enrich it with genre tags by joining
    data.csv (tracks) with data_w_genres.csv (artist → genres).

    The join key is the cleaned artist name.  Tracks whose artist is
    not found in data_w_genres get genre = 'Unknown'.

    The 'genres' column from data_w_genres contains a list of specific
    sub-genres (e.g. 'dark trap', 'new wave pop').  We map these to
    broad genre buckets via GENRE_MAP.

    Args:
        tracks_path : Override PATHS["tracks"].
        genres_path : Override PATHS["w_genres"].
        min_popularity, year_range, sample_n, random_state:
            Same as load_tracks().

    Returns:
        pd.DataFrame with an added 'genre' column.
    """
    tracks_path = tracks_path or PATHS["tracks"]
    genres_path = genres_path or PATHS["w_genres"]

    tracks = load_tracks(
        path=tracks_path,
        min_popularity=min_popularity,
        year_range=year_range,
        sample_n=sample_n,
        random_state=random_state,
    )

    wg = pd.read_csv(genres_path)
    wg["artists_clean"] = wg["artists"].apply(_clean_artists)

    wg["genre"] = wg["genres"].apply(
        lambda v: _map_to_broad_genre(_parse_list_str(v))
    )

    artist_genre = (
        wg[["artists_clean", "genre"]]
        .drop_duplicates("artists_clean")
        .rename(columns={"artists_clean": "artists"})
    )

    # Join
    tracks = tracks.merge(artist_genre, on="artists", how="left")
    tracks["genre"] = tracks["genre"].fillna("Unknown")

    print(f"[load_tracks_with_genres] Genre distribution:\n"
          f"{tracks['genre'].value_counts().to_string()}\n")
    return tracks


GENRE_MAP: list[tuple[str, str]] = [
    (r"hip.?hop|rap|trap|drill|grime",          "Hip-Hop"),
    (r"r&b|soul|funk|neo.soul",                 "R&B"),
    (r"pop",                                    "Pop"),
    (r"rock|punk|metal|grunge|emo|indie rock",  "Rock"),
    (r"indie|alternative|lo.?fi",               "Indie"),
    (r"jazz|blues|swing|bebop|bossa",           "Jazz"),
    (r"classical|orchestra|chamber|opera|piano concerto|symphony", "Classical"),
    (r"electronic|edm|techno|house|trance|dubstep|ambient|synth",  "Electronic"),
    (r"country|folk|americana|bluegrass",       "Country/Folk"),
    (r"latin|reggaeton|salsa|cumbia|bossa nova","Latin"),
    (r"reggae|dancehall|ska",                   "Reggae"),
]


def _map_to_broad_genre(sub_genres: list[str]) -> str:
    """
    Given a list of specific sub-genre strings, return the first
    matching broad genre label, or 'Other' if nothing matches.
    """
    for sg in sub_genres:
        sg_lower = sg.lower()
        for pattern, label in GENRE_MAP:
            if re.search(pattern, sg_lower):
                return label
    return "Other"


def load_by_genres(path: str | None = None) -> pd.DataFrame:
    """
    Load the genre-level aggregate dataset (data_by_genres.csv).

    Useful for EDA and radar/bar charts comparing genre audio profiles.

    Returns:
        pd.DataFrame — one row per genre, columns include all AUDIO_FEATURES
        plus 'popularity'.
    """
    path = path or PATHS["by_genres"]
    df   = pd.read_csv(path)
    df   = df.dropna(subset=AUDIO_FEATURES)
    print(f"[load_by_genres] {len(df):,} genre rows loaded.")
    return df


def load_by_year(path: str | None = None) -> pd.DataFrame:
    """
    Load the year-level aggregate dataset (data_by_year.csv).

    Useful for trend-over-time visualisations.

    Returns:
        pd.DataFrame — one row per year (1921–2020), columns include
        all AUDIO_FEATURES plus 'popularity'.
    """
    path = path or PATHS["by_year"]
    df   = pd.read_csv(path)
    df   = df.sort_values("year").reset_index(drop=True)
    print(f"[load_by_year] {len(df):,} year rows loaded.")
    return df


def load_by_artist(path: str | None = None) -> pd.DataFrame:
    """
    Load the artist-level aggregate dataset (data_by_artist.csv).

    Returns:
        pd.DataFrame — one row per artist, averaged audio features.
    """
    path = path or PATHS["by_artist"]
    df   = pd.read_csv(path)
    df["artists"] = df["artists"].apply(_clean_artists)
    print(f"[load_by_artist] {len(df):,} artist rows loaded.")
    return df


def build_clustering_pipeline(n_clusters: int = 8) -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler → KMeans.

    Scaling is essential because loudness (-60→0 dB) and tempo (BPM)
    would dominate the distance metric without it.

    Args:
        n_clusters: Number of K-Means clusters.

    Returns:
        sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
    ])


def fit_and_cluster(
    data: pd.DataFrame,
    n_clusters: int = 8,
) -> tuple[pd.DataFrame, Pipeline]:
    """
    Fit the clustering pipeline on audio features and append mood labels.

    Args:
        data       : Track-level DataFrame with AUDIO_FEATURES columns.
        n_clusters : Number of K-Means clusters.

    Returns:
        (DataFrame with 'cluster' and 'mood_cluster' columns, fitted Pipeline)
    """
    pipe = build_clustering_pipeline(n_clusters)
    X    = data[AUDIO_FEATURES]
    data = data.copy()
    data["cluster"]      = pipe.fit_predict(X)
    data["mood_cluster"] = data["cluster"].map(
        lambda c: CLUSTER_MOOD_LABELS.get(c, f"Cluster {c}")
    )
    print(f"[fit_and_cluster] {n_clusters} clusters fitted on {len(data):,} tracks.")
    return data, pipe


def compute_pca_embedding(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce 9 audio features to 2 principal components for scatter plots.

    Args:
        data: Track-level DataFrame with AUDIO_FEATURES columns.

    Returns:
        Same DataFrame with 'pc1' and 'pc2' columns added.
    """
    X       = data[AUDIO_FEATURES]
    scaler  = StandardScaler()
    X_std   = scaler.fit_transform(X)

    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(X_std)

    data    = data.copy()
    data["pc1"] = coords[:, 0]
    data["pc2"] = coords[:, 1]

    ev = pca.explained_variance_ratio_
    print(f"[PCA] Variance explained — PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%}")
    return data


def _normalise_feature(data: pd.DataFrame, feature: str) -> pd.Series:
    """Min-max normalise a feature column to [0, 1]."""
    col     = data[feature]
    mn, mx  = col.min(), col.max()
    return (col - mn) / (mx - mn + 1e-9)


def recommend_by_mood(
    mood: str,
    data: pd.DataFrame,
    top_n: int = 5,
    genre_filter: str | None = None,
) -> pd.DataFrame:
    """
    Recommend top_n songs whose audio features best match the given mood.

    Strategy: weighted L1 distance.  For each feature in the mood
    profile, compute |normalised_feature − target| × weight, sum
    across features, and return the songs with the lowest total score.

    Args:
        mood         : Mood key — one of MOOD_PROFILES keys.
        data         : Track-level DataFrame.
        top_n        : Number of recommendations to return.
        genre_filter : If set (e.g. 'Rock'), restrict recommendations
                       to that genre only.  Requires a 'genre' column.

    Returns:
        pd.DataFrame with columns: name, artists, genre, year,
        popularity, valence, energy, danceability, acousticness,
        match_score.

    Raises:
        ValueError: If mood is not recognised.
    """
    mood = mood.lower().strip()
    if mood not in MOOD_PROFILES:
        valid = ", ".join(MOOD_PROFILES)
        raise ValueError(f"Unknown mood '{mood}'. Valid options: {valid}")

    pool = data.copy()

    if genre_filter and "genre" in pool.columns:
        pool = pool[pool["genre"].str.lower() == genre_filter.lower()]
        if pool.empty:
            raise ValueError(f"No tracks found for genre '{genre_filter}'.")

    profile = MOOD_PROFILES[mood]
    score   = pd.Series(0.0, index=pool.index)

    for feature, (target, weight) in profile.items():
        norm   = _normalise_feature(pool, feature)
        score += weight * (norm - target).abs()

    pool["match_score"] = score

    # Build output column list (genre optional)
    base_cols = ["name", "artists", "year", "popularity",
                 "valence", "energy", "danceability", "acousticness", "match_score"]
    if "genre" in pool.columns:
        base_cols.insert(2, "genre")

    recommendations = (
        pool
        .nsmallest(top_n, "match_score")
        [base_cols]
        .reset_index(drop=True)
    )
    return recommendations


def genre_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mean audio features + popularity grouped by genre.

    Requires a 'genre' column (present when using load_tracks_with_genres).
    Falls back gracefully if the column is missing.
    """
    if "genre" not in data.columns:
        raise ValueError("DataFrame has no 'genre' column. "
                         "Use load_tracks_with_genres() instead of load_tracks().")
    return (
        data.groupby("genre")[AUDIO_FEATURES + ["popularity"]]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("popularity", ascending=False)
    )


def decade_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mean audio features grouped by decade, ordered chronologically.

    Works on any DataFrame that has a 'decade' column (load_tracks and
    load_tracks_with_genres both add it automatically).
    """
    present = [d for d in DECADE_NAMES if d in data["decade"].values]
    return (
        data.groupby("decade")[AUDIO_FEATURES]
        .mean()
        .round(3)
        .reindex(present)
        .dropna()
        .reset_index()
    )


def year_trend(by_year_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return the year-level aggregate data, optionally loading it fresh.

    Useful for line charts showing how audio features evolved over time.

    Args:
        by_year_df: Pre-loaded DataFrame from load_by_year().
                    If None, loads from PATHS["by_year"].

    Returns:
        pd.DataFrame sorted by year.
    """
    if by_year_df is None:
        by_year_df = load_by_year()
    return by_year_df.sort_values("year").reset_index(drop=True)


def top_genres_by_feature(
    by_genres_df: pd.DataFrame | None = None,
    feature: str = "danceability",
    top_n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return top_n genres ranked by a given audio feature.

    Args:
        by_genres_df : Pre-loaded DataFrame from load_by_genres().
        feature      : Column to rank by.
        top_n        : Number of genres to return.
        ascending    : Sort order (False = highest first).

    Returns:
        pd.DataFrame with columns: genres, <feature>.
    """
    if by_genres_df is None:
        by_genres_df = load_by_genres()
    if feature not in by_genres_df.columns:
        raise ValueError(f"Feature '{feature}' not in dataset.")
    return (
        by_genres_df[["genres", feature]]
        .sort_values(feature, ascending=ascending)
        .head(top_n)
        .reset_index(drop=True)
    )


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for audio features + popularity."""
    cols = AUDIO_FEATURES + ["popularity"]
    return data[cols].corr().round(3)


def feature_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics (count, mean, std, min, quartiles, max)
    for all audio features.
    """
    return data[AUDIO_FEATURES].describe().round(3)



def load_and_prepare(
    with_genres: bool = True,
    min_popularity: int = 10,
    year_range: tuple[int, int] = (1950, 2021),
    sample_n: int | None = None,
    n_clusters: int = 8,
    pca: bool = True,
) -> tuple[pd.DataFrame, Pipeline]:
    """
    One-call convenience function: load → cluster → (optionally) PCA.

    Args:
        with_genres    : If True, enrich with genre labels via data_w_genres.
        min_popularity : Drop tracks below this popularity score.
        year_range     : (min_year, max_year) inclusive.
        sample_n       : Randomly sample this many tracks (None = all).
        n_clusters     : Number of mood clusters for K-Means.
        pca            : If True, append pc1/pc2 PCA columns.

    Returns:
        (clustered DataFrame, fitted Pipeline)

    Example:
        >>> data, pipe = load_and_prepare(sample_n=20_000)
        >>> recs = recommend_by_mood("happy", data, top_n=5)
        >>> print(recs)
    """
    if with_genres:
        data = load_tracks_with_genres(
            min_popularity=min_popularity,
            year_range=year_range,
            sample_n=sample_n,
        )
    else:
        data = load_tracks(
            min_popularity=min_popularity,
            year_range=year_range,
            sample_n=sample_n,
        )

    data, pipe = fit_and_cluster(data, n_clusters=n_clusters)

    if pca:
        data = compute_pca_embedding(data)

    return data, pipe
