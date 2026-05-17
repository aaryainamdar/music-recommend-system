# 🎵 Music Mood Analyzer

A full-stack music analytics and recommendation app powered by real Spotify audio feature data. It clusters 170,000+ songs by mood using K-Means, visualizes audio feature patterns with PCA, and recommends tracks based on how you're feeling.

---

## Features

- **Mood-based recommendations** — pick a mood (happy, sad, energetic, calm, etc.) and get songs whose audio features best match it
- **K-Means clustering** — 8 mood clusters automatically discovered from audio features (valence, energy, danceability, etc.)
- **PCA scatter plot** — 9 audio dimensions reduced to 2D for visual exploration
- **Genre & decade summaries** — see how audio features differ across genres and evolve over time
- **Year trends** — line charts from 1920–2020 using year-level aggregates
- **Correlation matrix** — explore relationships between audio features
- **Feature distributions** — per-genre breakdown of any audio feature

---

## Project Structure

```
music-mood-analyzer/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── music_engine.py     # ML core: loading, clustering, PCA, recommendations
│   └── data/               # Dataset CSVs (see Dataset Setup below)
│       ├── data.csv
│       ├── data_w_genres.csv
│       ├── data_by_genres.csv
│       ├── data_by_year.csv
│       └── data_by_artist.csv
└── frontend/
    └── index.html          # Frontend (served by Flask)
```

---

## Dataset Setup

The project uses a real Spotify audio features dataset. Download the files and place them in `backend/data/`:

| File | Source zip | Rows |
|---|---|---|
| `data.csv` | `data_csv__1_.zip` | 170,653 tracks |
| `data_w_genres.csv` | `data_w_genres_csv.zip` | 28,680 artists + genres |
| `data_by_genres.csv` | *(plain CSV)* | 2,973 genre aggregates |
| `data_by_year.csv` | *(plain CSV)* | 100 year aggregates |
| `data_by_artist.csv` | `data_by_artist_csv.zip` | 28,680 artist aggregates |

After unzipping, run from the `backend/` folder:

```bash
mkdir data
move data.csv data\
move data_w_genres.csv data\
move data_by_genres.csv data\
move data_by_year.csv data\
move data_by_artist.csv data\
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/music-mood-analyzer.git
cd music-mood-analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask flask-cors pandas numpy scikit-learn
```

---

## Running the App

```bash
cd backend
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

On first boot the engine loads and samples 20,000 tracks, fits K-Means, and computes PCA — this takes about 5–10 seconds.

---

## API Reference

All endpoints are served at `http://127.0.0.1:5000`.

### `GET /api/stats`
Dataset overview — total songs, genre counts, decade counts, mood cluster counts, valid mood keys.

---

### `GET /api/genre-summary`
Mean audio features per broad genre (Rock, Pop, Hip-Hop, etc.).

---

### `GET /api/decade-trends`
Mean audio features grouped by decade (1950s–2020s).

---

### `GET /api/year-trends`
Year-by-year audio feature trends from `data_by_year.csv`.

| Query param | Type | Default | Description |
|---|---|---|---|
| `feature` | string | *(all)* | Return only one feature, e.g. `?feature=valence` |

---

### `GET /api/correlation`
Pearson correlation matrix for all audio features + popularity.

```json
{
  "labels": ["acousticness", "danceability", ...],
  "matrix": [[1.0, -0.3, ...], ...]
}
```

---

### `GET /api/pca`
2D PCA scatter data, sampled for performance.

| Query param | Type | Default | Description |
|---|---|---|---|
| `n` | int | 400 | Number of points to return |
| `genre` | string | *(all)* | Filter to one genre, e.g. `?genre=Rock` |

---

### `GET /api/feature-dist`
Distribution of one audio feature broken down by genre.

| Query param | Type | Default | Description |
|---|---|---|---|
| `feature` | string | `valence` | Audio feature name |

---

### `GET /api/top-genres`
Top genres from `data_by_genres.csv` ranked by an audio feature.

| Query param | Type | Default | Description |
|---|---|---|---|
| `feature` | string | `danceability` | Feature to rank by |
| `top_n` | int | 10 | Number of genres to return |
| `ascending` | bool | `false` | `true` for lowest-first |

---

### `GET /api/feature-stats`
Descriptive statistics (count, mean, std, min, quartiles, max) for all audio features.

---

### `POST /api/recommend`
Mood-based song recommendations.

**Request body:**
```json
{
  "mood": "happy",
  "top_n": 5,
  "genre": "Rock"
}
```

- `mood` — required. One of: `happy`, `sad`, `energetic`, `calm`, `angry`, `romantic`, `focus`, `party`
- `top_n` — optional, default `5`
- `genre` — optional filter. Omit to search all genres.

**Response:**
```json
{
  "mood": "happy",
  "genre_filter": "Rock",
  "recommendations": [
    {
      "name": "Mr. Brightside",
      "artists": "The Killers",
      "genre": "Rock",
      "year": 2003,
      "popularity": 87,
      "valence": 0.772,
      "energy": 0.930,
      "danceability": 0.536,
      "acousticness": 0.00491,
      "match_score": 0.124
    }
  ]
}
```

---

## Audio Features Explained

| Feature | Range | Description |
|---|---|---|
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `danceability` | 0–1 | How suitable the track is for dancing |
| `energy` | 0–1 | Perceptual intensity and activity |
| `instrumentalness` | 0–1 | Likelihood of no vocals |
| `liveness` | 0–1 | Presence of a live audience |
| `loudness` | -60–0 dB | Overall loudness |
| `speechiness` | 0–1 | Presence of spoken words |
| `tempo` | BPM | Overall estimated tempo |
| `valence` | 0–1 | Musical positiveness (high = happy, low = sad) |
| `popularity` | 0–100 | Spotify popularity score |

---

## Mood Profiles

Moods are matched by weighted distance across audio features:

| Mood | Key features targeted |
|---|---|
| `happy` | High valence, energy, danceability |
| `sad` | Low valence, low energy, high acousticness |
| `energetic` | Very high energy, danceability, tempo |
| `calm` | High acousticness, low energy, low liveness |
| `angry` | Very high energy, loudness; low valence |
| `romantic` | Moderate valence, acousticness, danceability |
| `focus` | High instrumentalness, low speechiness |
| `party` | Very high danceability, energy, valence |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask, Flask-CORS |
| ML | scikit-learn (K-Means, PCA, StandardScaler) |
| Data | pandas, NumPy |
| Dataset | Spotify dataset (Kaggle)- by Vatsal Mavani- https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset|

---

## License

MIT License — free to use, modify, and distribute.
