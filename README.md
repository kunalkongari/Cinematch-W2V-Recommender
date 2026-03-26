# 🎬 CineMatch — Word2Vec Movie Recommender

A production-ready, free-tier-deployable movie recommendation system powered by **Word2Vec** — trained directly on the TMDB movie corpus for domain-specific semantic understanding.

---

## 🧠 NLP Architecture

```
Raw movie data
     │
     ▼
Feature extraction: overview + genres + keywords + cast + director
     │
     ▼
Tokenization + Lemmatization (WordNetLemmatizer)
     │  overview tokens × 3  │  genres × 2  │  keywords × 2  │  director × 2
     ▼
Word2Vec Skip-gram training
  • vector_size = 300
  • window      = 5
  • epochs      = 15
  • trained on THIS corpus (domain-specific, not Wikipedia)
     │
     ▼
Mean-pool token vectors → one 300-dim embedding per movie
     │
     ▼
L2 normalization → pairwise cosine similarity matrix
     │
     ▼
Top-N recommendations
```

### Why Word2Vec over Bag-of-Words / TF-IDF+LSA

| | BoW (original) | TF-IDF + LSA | **Word2Vec (this)** |
|---|---|---|---|
| "warrior" ≈ "soldier" | ❌ | partial | ✅ |
| "space" ≈ "galaxy" | ❌ | partial | ✅ |
| Domain-specific training | ❌ | ❌ | ✅ movie corpus |
| RAM (free tier) | ~50 MB | ~80 MB | ~120 MB ✅ |
| Embedding quality | sparse | dense/generic | **dense/domain** |

---

## 📁 Project Structure

```
cinematch/
├── app.py              # Flask REST API
├── recommender.py      # Word2Vec NLP pipeline
├── train.py            # One-time training script
├── requirements.txt
├── Dockerfile          # Single-worker, free-tier optimised
├── render.yaml         # Render.com auto-deploy config
├── README_HF.md        # Hugging Face Spaces notes
└── templates/
    └── index.html      # Web UI
```

---

## ⚙️ Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download TMDB 5000 dataset from Kaggle:
#    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
#    Place tmdb_5000_movies.csv + tmdb_5000_credits.csv in project root

# 3. Train (~60s on a modern laptop)
python train.py

# 4. Run
python app.py
# → http://localhost:5000
```

---

## ☁️ Deployment (Free Tier)

### Render.com
1. Push repo to GitHub (commit `models/recommender.pkl` too)
2. New Web Service → connect GitHub repo
3. Render auto-detects `render.yaml` — done.

### Railway.app
```bash
railway login
railway init
railway up
```

### Hugging Face Spaces (Docker)
1. Create a new Space with SDK = Docker
2. Push this repo — HF builds from the `Dockerfile`
3. Set `PORT=7860` env var in Space settings
4. Commit your `models/recommender.pkl` (use HF Git LFS if > 10 MB)

### Docker (anywhere)
```bash
docker build -t cinematch .
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  cinematch
```

---

## 🌐 API

| Endpoint | Method | Description |
|---|---|---|
| `/api/recommend?title=Inception&n=10` | GET | Top-N similar movies |
| `/api/search?q=dark` | GET | Autocomplete titles |
| `/api/status` | GET | Model health check |
| `/api/build` | POST | Trigger training from API |

---

## 📄 License
MIT
