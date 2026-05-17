"""
app.py
------
Flask REST API — serves data from the music engine to the frontend.

Endpoints:
  GET  /api/stats          → dataset overview (genre counts, decade counts)
  GET  /api/genre-summary  → mean audio features per genre
  GET  /api/decade-trends  → mean audio features per decade
  GET  /api/year-trends    → audio feature trends by year (NEW)
  GET  /api/correlation    → audio feature correlation matrix
  GET  /api/pca            → PCA scatter data (sample of 400 songs)
  GET  /api/feature-dist   → feature distribution per genre
  GET  /api/top-genres     → top genres ranked by any audio feature (NEW)
  POST /api/recommend      → mood-based recommendations
                             body: {"mood": "happy", "top_n": 5, "genre": "Rock"}

Run:
  python app.py
  → http://127.0.0.1:5000
"""

import os
import sys

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from music_engine import (
    # loaders
    load_and_prepare,
    load_by_year,
    load_by_genres,
    # analysis
    genre_summary,
    decade_summary,
    year_trend,
    top_genres_by_feature,
    correlation_matrix,
    feature_stats,
    recommend_by_mood,
    # constants
    MOOD_PROFILES,
    AUDIO_FEATURES,
)

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────
#  Pre-load data once at startup
# ─────────────────────────────────────────────

print("[BOOT] Loading real Spotify dataset and fitting model …")

DATA, CLUSTER_PIPE = load_and_prepare(
    with_genres=True,       # join genre labels from data_w_genres.csv
    min_popularity=10,      # filter out near-zero popularity tracks
    year_range=(1950, 2021),
    sample_n=20_000,        # use 20k tracks — fast boot, still rich data
    n_clusters=8,
    pca=True,
)

# Load aggregate tables once — used by summary endpoints
BY_YEAR   = load_by_year()
BY_GENRES = load_by_genres()

print(f"[BOOT] Ready — {len(DATA):,} songs | "
      f"{DATA['genre'].nunique()} genres | "
      f"{DATA['year'].min()}–{DATA['year'].max()}\n")


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────

def _safe_json(obj):
    """Convert numpy scalars so Flask's jsonify doesn't choke."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ─────────────────────────────────────────────
#  Routes — Serve Frontend
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


# ─────────────────────────────────────────────
#  Routes — API
# ─────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    """High-level dataset overview."""
    return jsonify({
        "total_songs":   int(len(DATA)),
        "total_genres":  int(DATA["genre"].nunique()),
        "year_range":    [int(DATA["year"].min()), int(DATA["year"].max())],
        "genre_counts":  DATA["genre"].value_counts().to_dict(),
        "decade_counts": DATA["decade"].value_counts().to_dict(),
        "mood_counts":   DATA["mood_cluster"].value_counts().to_dict(),
        "valid_moods":   list(MOOD_PROFILES.keys()),
        "audio_features": AUDIO_FEATURES,
    })


@app.route("/api/genre-summary")
def api_genre_summary():
    """Mean audio features per broad genre."""
    df = genre_summary(DATA)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/decade-trends")
def api_decade_trends():
    """Audio feature trends across decades (from track-level data)."""
    df = decade_summary(DATA)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/year-trends")
def api_year_trends():
    """
    Audio feature trends by individual year (from data_by_year.csv).
    Finer-grained than decade trends — good for line charts.

    Query params:
      feature (str): single feature to return, e.g. ?feature=valence
                     omit to return all features.
    """
    feature = request.args.get("feature")
    df = year_trend(BY_YEAR)

    if feature:
        if feature not in AUDIO_FEATURES + ["popularity"]:
            return jsonify({"error": f"Unknown feature '{feature}'"}), 400
        df = df[["year", feature]]

    return jsonify(df.to_dict(orient="records"))


@app.route("/api/correlation")
def api_correlation():
    """Pearson correlation matrix for audio features + popularity."""
    corr = correlation_matrix(DATA)
    return jsonify({
        "labels": list(corr.columns),
        "matrix": corr.values.tolist(),
    })


@app.route("/api/pca")
def api_pca():
    """
    PCA 2-D scatter — sampled for frontend performance.

    Query params:
      n    (int): sample size, default 400
      genre(str): filter to one genre, e.g. ?genre=Rock
    """
    n     = int(request.args.get("n", 400))
    genre = request.args.get("genre")

    sample = DATA.copy()
    if genre:
        sample = sample[sample["genre"].str.lower() == genre.lower()]

    sample = sample.sample(n=min(n, len(sample)), random_state=42)

    records = sample[[
        "pc1", "pc2", "genre", "mood_cluster",
        "name", "artists", "popularity", "valence", "energy",
    ]].rename(columns={"mood_cluster": "mood"}).to_dict(orient="records")

    return jsonify(records)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Return top-n mood-matched songs.

    Body (JSON):
      mood   (str) : one of the valid mood keys, e.g. "happy"
      top_n  (int) : number of results, default 5
      genre  (str) : optional genre filter, e.g. "Rock"
    """
    body  = request.get_json(force=True) or {}
    mood  = body.get("mood", "happy")
    top_n = int(body.get("top_n", 5))
    genre = body.get("genre")          # NEW — optional genre filter

    try:
        recs = recommend_by_mood(mood, DATA, top_n=top_n, genre_filter=genre)
        return jsonify({
            "mood":            mood,
            "genre_filter":    genre,
            "recommendations": recs.to_dict(orient="records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/feature-dist")
def api_feature_dist():
    """
    Distribution of one audio feature broken down by genre.

    Query params:
      feature (str): audio feature name, default "valence"
    """
    feature = request.args.get("feature", "valence")
    if feature not in AUDIO_FEATURES:
        return jsonify({"error": f"Unknown feature '{feature}'"}), 400

    result = {}
    for genre, group in DATA.groupby("genre"):
        vals = group[feature].round(4).tolist()
        result[genre] = {
            "values": vals[:200],
            "mean":   round(float(group[feature].mean()), 3),
            "std":    round(float(group[feature].std()),  3),
        }
    return jsonify({"feature": feature, "data": result})


@app.route("/api/top-genres")
def api_top_genres():
    """
    Top genres from data_by_genres.csv ranked by an audio feature.

    Query params:
      feature   (str) : audio feature to rank by, default "danceability"
      top_n     (int) : number of genres to return, default 10
      ascending (bool): "true" for lowest-first, default false (highest-first)
    """
    feature   = request.args.get("feature", "danceability")
    top_n     = int(request.args.get("top_n", 10))
    ascending = request.args.get("ascending", "false").lower() == "true"

    try:
        df = top_genres_by_feature(BY_GENRES, feature=feature,
                                   top_n=top_n, ascending=ascending)
        return jsonify(df.to_dict(orient="records"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/feature-stats")
def api_feature_stats():
    """Descriptive statistics (mean, std, min, max, quartiles) for all audio features."""
    stats_df = feature_stats(DATA)
    return jsonify(stats_df.to_dict())


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   Music Mood Analyzer  — Flask API   ║")
    print("║   http://127.0.0.1:5000              ║")
    print("╚══════════════════════════════════════╝\n")
    app.run(debug=True, port=5000)