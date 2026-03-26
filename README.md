# 🎬 CineMatch — AI-Powered Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![Word2Vec](https://img.shields.io/badge/NLP-Word2Vec-orange?style=flat)
![Deployed](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> **Live Demo → [https://cinematch-w2v-recommender.onrender.com](https://cinematch-w2v-recommender.onrender.com)**

A production-deployed, content-based movie recommendation system that uses **domain-trained Word2Vec embeddings** and **cosine similarity** to find semantically similar movies. Type any movie title and get 10 intelligent recommendations — complete with posters, descriptions, cast, and match scores.

---

## 📌 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [Evolution: From Notebook to Production](#-evolution-from-notebook-to-production)
3. [How It Works — Full NLP Pipeline](#-how-it-works--full-nlp-pipeline)
4. [Why Word2Vec Over Other Techniques](#-why-word2vec-over-other-techniques)
5. [Project Structure](#-project-structure)
6. [Tech Stack — What & Why](#-tech-stack--what--why)
7. [Dataset](#-dataset)
8. [API Reference](#-api-reference)
9. [Running Locally](#-running-locally)
10. [Deployment on Render](#-deployment-on-render)
11. [TMDB Poster Integration](#-tmdb-poster-integration)
12. [Known Limitations & Future Work](#-known-limitations--future-work)

---

## 🎯 What This Project Does

CineMatch is a **content-based movie recommendation engine**. Given any movie title, it returns the 10 most similar movies based on the semantic meaning of the movie's plot, genre, keywords, cast, and director.

**Content-based filtering** means the system understands *what a movie is about*, not *who watched it*. There are no user accounts, no ratings history, and no collaborative data needed. This makes it work perfectly even for new users or obscure films.

### Example

You search for **"Inception"** → the system returns movies that are semantically similar:
- Films with psychological / mind-bending plots
- Heist or thriller themes
- Christopher Nolan's directorial style
- Sci-fi concepts like dreams or alternate realities

This works because the system has learned that words like *"dream"*, *"subconscious"*, *"heist"*, and *"reality"* cluster together in the same semantic space.

---

## 🔄 Evolution: From Notebook to Production

This project started as a basic Jupyter notebook using **Bag-of-Words** and was completely rebuilt into a deployable web application with modern NLP. Here is the full journey:

### Stage 1 — Original Notebook (Bag-of-Words)

The original code used:
- `CountVectorizer` — counts how many times each word appears in a document
- `PorterStemmer` — crudely reduces words to a root form (`loving → love`)
- `cosine_similarity` computed on a raw sparse 5000-feature matrix
- No web interface — just `print()` output in a Jupyter notebook

**The fundamental problem with Bag-of-Words:** It treats every word as a completely independent token with no relationship to any other word. The words `"warrior"` and `"soldier"` are entirely different in a BoW model — so *Gladiator* would never match *Braveheart* on those words, even though they describe essentially the same concept. There is zero semantic understanding.

### Stage 2 — Intermediate Upgrade (TF-IDF + LSA)

Upgraded to:
- `TfidfVectorizer` — weighs rare, informative words higher than common words. The word *"heist"* appearing in 5 movies is more meaningful than *"man"* appearing in 3000 movies.
- `TruncatedSVD` (Latent Semantic Analysis / LSA) — compresses the sparse 10,000-feature TF-IDF matrix down to 200 dense dimensions, capturing some statistical co-occurrence relationships between words
- Lemmatization instead of stemming for cleaner word normalization

**Improvement:** LSA can discover that movies using the words `"galaxy"`, `"spaceship"`, and `"alien"` are related to movies using `"cosmos"`, `"rocket"`, and `"extraterrestrial"` — purely through statistical co-occurrence. But this is still a shallow, mathematical relationship, not true semantic understanding.

### Stage 3 — Current: Word2Vec + Full Web Application

The final production version uses:
- **Domain-trained Word2Vec** — a neural network that genuinely learns the meaning of words from context, trained specifically on movie language
- **Mean-pooled document vectors** — each movie is represented as a single 300-dimensional dense semantic vector
- **Flask REST API** — all functionality exposed as clean HTTP endpoints
- **Cinematic web UI** — dark-themed, responsive frontend with autocomplete, animated cards, and similarity scores
- **TMDB poster integration** — real movie posters and descriptions fetched live
- **Production deployment** — live on Render.com with gunicorn WSGI server, accessible globally

---

## 🧠 How It Works — Full NLP Pipeline

Here is a step-by-step walkthrough of every transformation that happens when you run `python train.py`:

### Step 1: Data Loading & Merging

```
tmdb_5000_movies.csv  ──┐
                         ├──► merge on "title" ──► unified DataFrame (~4806 movies)
tmdb_5000_credits.csv ──┘
```

Two CSV files are loaded and merged on the `title` column. The movies file contains plot details, ratings, and genres. The credits file contains cast and crew information stored as JSON strings inside CSV cells. After merging, only the columns relevant to recommendation are kept: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`, `vote_average`, `vote_count`, `popularity`, `release_date`.

### Step 2: Feature Extraction from JSON Columns

The raw CSV data has JSON-encoded strings inside cells (e.g. the `genres` column looks like `[{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}]`). Python's `ast.literal_eval()` safely parses these strings into actual Python objects, then only the useful parts are extracted:

| Column | Raw Format | What We Extract |
|---|---|---|
| `genres` | `[{"id":28,"name":"Action"},...]` | All genre names: `["Action","Thriller"]` |
| `keywords` | `[{"id":1,"name":"spy"},...]` | All keyword names: `["spy","assassin","heist"]` |
| `cast` | `[{"name":"Leonardo DiCaprio",...},...]` | **Top 3 actors only** |
| `crew` | `[{"job":"Director","name":"Nolan"},...]` | **Director name only** |

**Why top 3 cast only?** Movies can have dozens of credited cast members. Taking only the top 3 keeps the signal strong and focused on the leads. Including supporting actors and extras would add noise — a minor actor appearing in two completely different types of films would create false similarity between them.

**Why director only from crew?** The director has the single greatest influence on a film's visual style, pacing, tone, and overall feel. Other crew roles (cinematographer, editor, composer) are meaningful but less discriminating for the purposes of recommendation. Keeping only the director keeps the signal clean.

### Step 3: Name Collision Prevention

Multi-word names like `"Sam Worthington"` are collapsed into single tokens: `"SamWorthington"`. This is a critical preprocessing step that prevents token ambiguity.

```python
def _clean_name(name):
    return name.replace(" ", "")

# "Sam Worthington" → "SamWorthington"
# "James Cameron"   → "JamesCameron"
# "Science Fiction" → "ScienceFiction"
```

Without this step, the word `"Sam"` from actor Sam Worthington could accidentally influence similarity with completely unrelated movies that happen to use the word `"Sam"` in their plot overview (e.g. *The Lord of the Rings*, where Samwise Gamgee is a major character). Joining the parts into one token makes the name a unique, unambiguous identifier.

### Step 4: Text Normalization Pipeline

Every text field passes through a four-stage cleaning pipeline:

**1. Lowercase conversion**
```
"Action Thriller" → "action thriller"
```
Ensures `"Action"` and `"action"` are treated as the same word.

**2. Punctuation removal**
```
"Spider-Man: Homecoming" → "spider man  homecoming"
```
A regex `[^a-z\s]` strips everything that isn't a lowercase letter or whitespace. Hyphens, colons, and apostrophes are removed.

**3. Stopword removal**
Common English words that carry no semantic meaning for recommendation are removed using NLTK's English stopword list: `"the"`, `"a"`, `"an"`, `"is"`, `"are"`, `"was"`, `"were"`, `"in"`, `"on"`, `"at"`, `"to"`, `"for"` etc. These words appear in almost every movie's overview and would create false similarity between completely unrelated films.

**4. Lemmatization**
Words are reduced to their dictionary base form using NLTK's `WordNetLemmatizer`:
```
"running"  → "run"
"movies"   → "movie"
"warriors" → "warrior"
"loved"    → "love"
"galaxies" → "galaxy"
```

**Why lemmatization instead of stemming?** Intially used Porter Stemming, which applies crude suffix-stripping rules without understanding word meaning. It produces non-words: `"universe"` → `"univers"`, `"caring"` → `"car"`. These broken tokens degrade Word2Vec quality because the model cannot learn meaningful relationships for non-words. Lemmatization uses a real dictionary (WordNet) and understands grammar, so it always produces valid English words that Word2Vec can learn better representations for.

### Step 5: Weighted Token Corpus Construction

Each movie is converted into a **flat list of tokens** (called a "document") that will be fed to Word2Vec for training. Crucially, different feature fields are repeated different numbers of times to signal their relative importance:

```python
def movie_tokens(row):
    overview_tokens  = tokenize(row["overview"])
    genre_tokens     = [clean_name(g) for g in row["genres"]]
    keyword_tokens   = [clean_name(k) for k in row["keywords"]]
    cast_tokens      = [clean_name(c) for c in row["cast"]]
    director_tokens  = [clean_name(d) for d in row["director"]]

    return (
        overview_tokens  * 3 +   # plot description — most important
        genre_tokens     * 2 +   # genre — very important
        keyword_tokens   * 2 +   # TMDB keywords — very important
        cast_tokens      * 1 +   # lead actors — moderately important
        director_tokens  * 2     # director — very important
    )
```

**Why repeat tokens?** Word2Vec learns word relationships by observing how often words appear near each other (within the context window). By repeating the overview tokens 3 times, plot-related words appear much more frequently in each movie's document. This means they will have more influence over the final movie embedding. This is a form of **manual feature weighting** that doesn't require a separate weighting mechanism — we exploit the training dynamics of Word2Vec itself.

For a movie like *Inception*: the words `"dream"`, `"subconscious"`, `"heist"` from the overview appear 3 times each, while `"sciencefiction"` appears twice and `"LeonardoDiCaprio"` appears once.

### Step 6: Word2Vec Training (Skip-gram Architecture)

A Word2Vec model is trained entirely **from scratch** on the movie corpus using the **skip-gram** architecture:

```python
w2v = Word2Vec(
    sentences   = corpus,     # 4806 token lists, one per movie
    vector_size = 300,        # each word → 300-dimensional dense vector
    window      = 5,          # consider 5 words left and right as context
    min_count   = 2,          # discard words appearing only once (noise)
    sg          = 1,          # 1 = skip-gram architecture
    epochs      = 15,         # 15 complete passes over the training data
    workers     = 4,          # parallel CPU threads for faster training
    seed        = 42,         # random seed for reproducibility
)
```

**What is Word2Vec?**
Word2Vec is a shallow neural network trained on a text corpus. The skip-gram variant is given a word and asked to predict what words appear around it in the training data. Through this prediction task, the network learns rich vector representations of every word such that words with similar meanings end up close together in the 300-dimensional vector space:

```
cosine_similarity( vector("warrior"),  vector("soldier")  ) ≈ 0.89
cosine_similarity( vector("space"),    vector("galaxy")   ) ≈ 0.91
cosine_similarity( vector("romance"),  vector("love")     ) ≈ 0.87
cosine_similarity( vector("warrior"),  vector("romance")  ) ≈ 0.12
```

This means a movie about "space warriors" will end up similar to a movie about "galaxy soldiers" — even if they share no words at all — because the word vectors for these synonymous concepts are geometrically close.

**Why skip-gram (`sg=1`) over CBOW (`sg=0`)?**
CBOW (Continuous Bag of Words) predicts a target word from its surrounding context. Skip-gram does the opposite: it predicts the surrounding context words from a single target word. Skip-gram is known to work better on smaller datasets and is particularly good at representing **rare words** — both important here, since our corpus is only ~4800 documents and movie-specific terms (director names, franchise-specific words, character names) appear infrequently.

**Why train on THIS corpus instead of using a pretrained model?**
Pre-trained Word2Vec models (Google News, Wikipedia) would know general word relationships, but they wouldn't understand domain-specific movie language. In a pretrained model, `"Nolan"` is just a surname. In our domain-trained model, `"Nolan"` (from JamesCameron and ChristopherNolan appearing together in sci-fi/action context) develops a vector close to `"villeneuve"` (another cerebral sci-fi director) and `"kubrick"`. The model learns the *movie industry meaning* of these names.

### Step 7: Movie Embedding via Mean Pooling

After Word2Vec training, every word in the vocabulary has a 300-dimensional vector. To get a single vector representing an entire movie, we **mean pool** all token vectors:

```python
def mean_vector(tokens):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0)   # element-wise average across all token vectors

movie_embedding = mean_vector(movie_tokens)   # shape: (300,)
```

This produces a single 300-dimensional vector per movie. Movies with similar plots, overlapping genres, shared directors, and common themes will have vectors that point in similar directions in this 300-dimensional space.

The full embedding matrix has shape `(4806, 300)` — 4806 movies, each with a 300-dimensional vector.

### Step 8: L2 Normalization

All movie vectors are L2-normalized (each vector is divided by its own magnitude so it has unit length = 1.0):

```python
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings)   # each row now has magnitude 1.0
```

After this step, **cosine similarity equals the dot product**. This is a useful mathematical identity that makes the subsequent similarity computation faster, since dot products are highly optimized in NumPy.

### Step 9: Pairwise Cosine Similarity Matrix

A complete pairwise cosine similarity matrix is computed across all movies:

```python
similarity = cosine_similarity(embeddings)
# shape: (4806, 4806)
# similarity[i][j] = how similar movie i is to movie j
# value range: 0.0 (nothing in common) to 1.0 (identical)
```

This 4806 × 4806 matrix is computed **once** during training and saved to disk. At query time, looking up recommendations for a movie is just:
1. Find the movie's row index
2. Read that row of the similarity matrix
3. Sort by score descending
4. Return the top 10 movies (skipping the movie itself at position 0)

This is an **O(n log n)** operation and takes milliseconds, regardless of how complex the original NLP pipeline was.

### Step 10: Saving the Model

The trained DataFrame (with all movie metadata) and the similarity matrix are serialized together using Python's `pickle` module:

```python
pickle.dump({"df": self.df, "similarity": self.similarity}, f)
# → models/recommender.pkl  (typically 50-100 MB)
```

When the Flask server starts, it loads this file once into memory. All subsequent recommendation requests are served from the in-memory data with no disk access or recomputation.

---

## 🆚 Why Word2Vec Over Other Techniques

| Technique | Semantic Understanding | Needs Training | RAM Usage | Free Hostable | Recommendation Quality |
|---|---|---|---|---|---|
| Bag-of-Words | None | No | ~30 MB | Yes | Basic |
| TF-IDF | None (just weights) | No | ~60 MB | Yes | OK |
| TF-IDF + LSA | Statistical only | No | ~80 MB | Yes | Good |
| **Word2Vec** | Neural, domain-specific | On corpus | ~150 MB | Yes | Better |
| Sentence-BERT | Deep contextual | Pretrained | ~2-4 GB | No | Best |
| OpenAI Embeddings | Deep contextual | API call | ~Minimal | + cost | Best |

**Word2Vec was chosen** because it provides genuine semantic understanding while remaining deployable on free-tier cloud infrastructure. The free tier on Render provides 512 MB RAM — Sentence-BERT alone requires 1.5-2 GB minimum, making it impossible to host for free.

Additionally, Word2Vec **trained specifically on the movie corpus** outperforms generic pretrained models for this specific task, because it learns domain-specific semantic relationships (director styles, genre conventions, franchise connections) that general-purpose models trained on Wikipedia or news articles don't capture.

---

## 📁 Project Structure

```
cinematch/
│
├── app.py                    # Flask web server + REST API (5 endpoints)
├── recommender.py            # Core NLP engine (entire Word2Vec pipeline)
├── train.py                  # Standalone CLI training script
│
├── templates/
│   └── index.html            # Complete frontend (HTML + CSS + JS, single file)
│
├── models/
│   └── recommender.pkl       # Serialized trained model (generated by train.py)
│
├── requirements.txt          # Pinned Python dependencies
├── Dockerfile                # Docker container configuration
├── render.yaml               # Render.com auto-deployment config
├── .python-version           # Pins Python 3.11.8 (prevents Render using 3.14)
└── .gitignore
```

### File Responsibilities

**`recommender.py`** is the brain. It contains the `MovieRecommender` class:
- `build(movies_csv, credits_csv)` — runs the complete NLP pipeline end-to-end and saves the model
- `load()` — deserializes a pretrained model from `models/recommender.pkl`
- `recommend(title, n)` — given a movie title, returns the top-N most similar movies
- `search(query, limit)` — partial title search for autocomplete functionality

**`app.py`** is the server. It creates the Flask application, loads the trained model on startup, and defines 5 HTTP endpoints. It also contains the TMDB poster proxy logic.

**`train.py`** is a convenience script. Running `python train.py` triggers the full pipeline — loading CSVs, training Word2Vec, building the similarity matrix, and saving the model. Run this once before deploying.

**`templates/index.html`** is the entire frontend in a single self-contained file. Uses vanilla HTML, CSS, and JavaScript — no React, no Vue, no build tools. Features: animated card grid, real-time autocomplete with keyboard navigation, lazy-loaded movie posters with fade-in animation, similarity score progress bars, and a film-themed dark aesthetic with grain texture.

---

## 🛠 Tech Stack — What & Why

### Backend Libraries

**Flask 3.0.3** — Chosen as the web framework because it is extremely lightweight and adds minimal overhead for a simple serving use case. A full framework like Django would be over-engineered here. Flask is also the standard choice in the Python ML community for serving models.

**Gensim 4.3.2** — The industry-standard library for Word2Vec. Gensim's implementation uses highly optimized C extensions under the hood, making training significantly faster than a pure Python implementation. It also provides a clean, well-documented API.

**scikit-learn 1.4.2** — Used for two specific functions: `cosine_similarity()` (computing the pairwise similarity matrix) and `normalize()` (L2 normalization of embeddings). Both are NumPy-backed and extremely fast.

**pandas 2.2.2** — Essential for loading, merging, and manipulating the two CSV files. The `apply()` method makes feature engineering clean and readable.

**NumPy 1.26.4** — Foundation for all matrix operations. The 4806 × 4806 similarity matrix is a NumPy ndarray; mean pooling and normalization use NumPy operations.

**NLTK 3.8.1** — Used for two components: `WordNetLemmatizer` (dictionary-based lemmatization) and the English stopword corpus. Has graceful fallback — if NLTK downloads fail (e.g. in restricted network environments), the code continues with basic text cleaning.

**Gunicorn 21.2.0** — Production-grade WSGI (Web Server Gateway Interface) server. Flask's built-in development server is single-threaded and not safe for production. Gunicorn is the standard way to deploy Flask apps.

### Frontend

The frontend is intentionally built with zero JavaScript frameworks. Every feature is implemented in ~150 lines of vanilla JS:
- **Fetch API** for all HTTP calls (async/await pattern)
- **DOM manipulation** for rendering cards dynamically
- **CSS animations** (`@keyframes fadeUp`) for staggered card entrance
- **CSS custom properties** (variables) for consistent theming
- **Google Fonts CDN** for typography (`Bebas Neue` display font, `DM Sans` body font)

Keeping the frontend framework-free means there is no build pipeline, no `node_modules`, no webpack configuration — just one HTML file served directly by Flask.

### Infrastructure

**Render.com** — Chosen for deployment because it offers a genuinely free tier for web services, supports Python natively (no Docker required), and auto-deploys from GitHub on every push. The `render.yaml` file in the repository means zero manual configuration on the dashboard.

**GitHub** — Source control and the deployment trigger. Every `git push` to the `main` branch automatically triggers a new Render build.

**TMDB API** — The Movie Database provides a free API (up to 1000 requests per day on the developer tier) for fetching movie posters, backdrops, and additional metadata. The API key is stored as a server-side environment variable, never exposed to the browser.

---

## 📊 Dataset

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle

This dataset was collected from [The Movie Database (TMDB)](https://www.themoviedb.org/) and contains metadata for approximately 5,000 films, primarily from the last few decades up to 2017.

### Files

**`tmdb_5000_movies.csv`** (contains per-movie metadata):
- `movie_id` — unique TMDB numeric identifier
- `title` — official movie title
- `overview` — plot synopsis (typically 1-3 paragraphs of natural language)
- `genres` — JSON array of genre objects (e.g. Action, Drama, Sci-Fi)
- `keywords` — JSON array of descriptive keyword objects curated by TMDB editors
- `vote_average` — average user rating on TMDB (scale 0-10)
- `vote_count` — total number of user votes
- `popularity` — TMDB's proprietary popularity score
- `release_date` — release date as a string

**`tmdb_5000_credits.csv`** (contains cast and crew):
- `title` — movie title (used as the merge key)
- `cast` — JSON array of cast members with names, character names, and order of billing
- `crew` — JSON array of crew members with names and job titles

### After Processing

After merging on `title`, dropping rows with missing overviews, and running feature extraction, the final usable dataset is approximately **4,806 movies**.

---

## 🌐 API Reference

The Flask server exposes 5 REST endpoints:

### `GET /`
Serves the main web UI (renders `templates/index.html`).

---

### `GET /api/recommend`

Returns the top-N semantically similar movies for a given title.

**Query Parameters:**

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `title` | string | ✅ Yes | — | Must be an exact title from the dataset |
| `n` | integer | No | 10 | Max 20 results |

**Example:**
```
GET /api/recommend?title=Inception&n=5
```

**Success Response (200):**
```json
{
  "query": "Inception",
  "results": [
    {
      "title": "Interstellar",
      "movie_id": 157336,
      "score": 0.9124,
      "genres": ["Adventure", "Drama", "Science Fiction"],
      "overview": "Interstellar chronicles the adventures of a group of explorers...",
      "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
      "director": ["Christopher Nolan"],
      "rating": 8.1,
      "vote_count": 11000,
      "year": "2014"
    }
  ]
}
```

**Error Responses:**
- `400 Bad Request` — `title` parameter not provided
- `404 Not Found` — movie title not found in the dataset
- `503 Service Unavailable` — model has not been loaded yet

---

### `GET /api/search`

Real-time autocomplete — returns movie titles that contain the query string.

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Partial title query (minimum 2 characters) |

**Example:**
```
GET /api/search?q=dark
→ ["The Dark Knight", "Dark Shadows", "The Dark Knight Rises", "Dark City"]
```

Returns an empty JSON array `[]` if no matches found or if the query is less than 2 characters.

---

### `GET /api/status`

Health check — returns whether the model is loaded and how many movies are indexed.

**Response:**
```json
{
  "ready": true,
  "movie_count": 4806
}
```

Used by the frontend on page load to show the green status indicator and movie count in the header, or to show the "model not loaded" warning banner.

---

### `GET /api/poster`

Proxy to the TMDB API — fetches poster image URL, backdrop URL, and full plot overview. Returns `null` values gracefully if `TMDB_API_KEY` environment variable is not set.

| Parameter | Type | Description |
|---|---|---|
| `movie_id` | integer | TMDB movie ID (preferred — most accurate) |
| `title` | string | Movie title (fallback text search) |

**Response:**
```json
{
  "poster":   "https://image.tmdb.org/t/p/w342/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
  "backdrop": "https://image.tmdb.org/t/p/w780/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
  "overview": "A thief who steals corporate secrets through the use of dream-sharing technology..."
}
```

---

### `POST /api/build`

Triggers model training from within the running application. Useful for retraining without restarting the server.

**Request Body:**
```json
{
  "movies_csv":  "tmdb_5000_movies.csv",
  "credits_csv": "tmdb_5000_credits.csv"
}
```

**Response:**
```json
{
  "status": "ok",
  "movies_indexed": 4806
}
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11 (other versions may work but 3.11 is tested)
- pip
- The TMDB 5000 dataset CSV files (downloaded from Kaggle)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Place dataset CSV files in the project root directory
#    cinematch/
#    ├── tmdb_5000_movies.csv   ← here
#    ├── tmdb_5000_credits.csv  ← here
#    └── ...

# 5. Train the model (run once — takes ~60 seconds)
python train.py

# Expected output:
# Loading CSVs...
# Extracting features...
# Training Word2Vec (skip-gram, 300-dim, 15 epochs)...
# Building movie embedding matrix...
# Computing pairwise cosine similarity...
# Done — 4806 movies indexed. Model saved to models/recommender.pkl

# 6. Start the Flask development server
python app.py
```

Open **http://localhost:5000** in your browser.

### Enable Movie Posters (Optional)

Get a free API key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) (Developer Plan, instant approval), then:

```bash
# Mac / Linux
TMDB_API_KEY=your_key_here python app.py

# Windows Command Prompt
set TMDB_API_KEY=your_key_here
python app.py

# Windows PowerShell
$env:TMDB_API_KEY="your_key_here"
python app.py
```

### Common Issues

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: gensim` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `FileNotFoundError: tmdb_5000_movies.csv` | CSV files not in project root | Move files to same folder as `app.py` |
| `"Model not loaded"` banner in browser | `train.py` not run yet | Run `python train.py` first |
| Port 5000 already in use | Another process using port 5000 | Run `PORT=8080 python app.py` |

---

## ☁️ Deployment on Render

### Why Render

Render was chosen because it supports Python natively, offers a genuinely free tier, auto-deploys from GitHub, and the `render.yaml` configuration file makes setup effortless. No Docker knowledge required.

### Deployment Steps

**1. Push code to GitHub**
```bash
git init
git add .
git add -f models/recommender.pkl    # force-add (models/ is gitignored)
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/cinematch.git
git push -u origin main
```

**2. Connect to Render**
- Sign up at [render.com](https://render.com) using GitHub
- New → Web Service → connect your repository
- Render reads `render.yaml` automatically — no manual config needed

**3. Add environment variables**
In Render Dashboard → Environment tab → add:
```
TMDB_API_KEY = your_tmdb_key_here
```

**4. Deploy** — Render builds and starts the app. Live in ~3 minutes.

### Configuration Files Explained

**`render.yaml`**
```yaml
services:
  - type: web
    name: cinematch-movie-recommender
    env: python
    buildCommand: |
      pip install -r requirements.txt &&
      python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
    startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.8
```

The `--preload` flag tells gunicorn to load the application (and therefore the model file) **before** forking worker processes. This means the 100 MB model is loaded once and shared across workers via OS copy-on-write, rather than each worker loading its own copy.

**`.python-version`** — Contains `3.11.8`. This was critical: without this file, Render defaulted to Python 3.14 (an alpha/beta release), which caused pandas to fail during C extension compilation — the exact error in the build logs. Pinning to 3.11.8 resolved the issue completely.

**`requirements.txt` with pinned versions** — Using `pandas==2.2.2` instead of `pandas>=2.0.0` prevents surprise build failures when new package versions with breaking changes are released.

### Free Tier Architecture

```
GitHub push to main
        │
        ▼
Render detects change via webhook
        │
        ▼
Build phase (~2-3 min):
  pip install -r requirements.txt
  nltk download wordnet + stopwords
        │
        ▼
Start phase:
  gunicorn --preload loads app.py
  app.py loads models/recommender.pkl (~100 MB) into RAM
        │
        ▼
App live: https://cinematch-w2v-recommender.onrender.com
```

**Free tier note:** The app sleeps after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake the server. Subsequent requests are instant.

---

## 🎨 TMDB Poster Integration

Movie posters and rich descriptions are fetched from [The Movie Database API](https://developers.themoviedb.org/3). The `/api/poster` endpoint acts as a **server-side proxy** between the browser and TMDB.

### Why a Server-Side Proxy?

Calling the TMDB API directly from JavaScript (in the browser) would expose the API key in the page source — anyone could view it with browser developer tools. By routing the request through the Flask server, the API key stays in the server environment and is never sent to the browser.

### Lookup Strategy

The endpoint tries two approaches in order:
1. **By movie ID** (preferred) — uses the TMDB numeric ID stored in our dataset. IDs are unique and permanent, so this always returns the correct movie.
2. **By title search** (fallback) — if no ID is available, sends a text search query to TMDB and takes the first result.

### Graceful Degradation

If `TMDB_API_KEY` is not set, the endpoint returns `{"poster": null, "backdrop": null, "overview": null}` and the frontend shows a film icon placeholder. The core recommendation system works completely independently of the poster API — posters are an enhancement, not a requirement.

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

**Exact title matching** — The recommendation engine requires a title that exactly matches one in the dataset. The autocomplete search widget is provided to help users find valid titles, but there is no fuzzy matching for typos.

**Static 2017 dataset** — The model is trained on a TMDB snapshot containing films up to approximately 2017. Movies released after that (Dune, Oppenheimer, Everything Everywhere All at Once, etc.) are not in the dataset and cannot be searched or recommended.

**Mean pooling loses word order** — Converting a movie's token list to a single vector via mean pooling creates a "bag of vectors" — positional/sequential information is lost. The sentence *"hero saves the world"* and *"world destroys the hero"* would produce very similar vectors. For this use case (keyword-dense metadata rather than natural prose), this limitation has minimal practical impact.

**Cold start on free hosting** — Render's free tier sleeps after inactivity, causing ~30 second delays on the first request.

### Potential Future Improvements

- **Sentence-BERT upgrade** — Replace Word2Vec with `sentence-transformers` for genuinely deeper semantic understanding of plot overviews. Requires hosting with ≥4 GB RAM.
- **Newer / live dataset** — Integrate the TMDB API to fetch current movie data rather than using a static 2017 CSV.
- **Hybrid recommendations** — Combine content-based (this system) with collaborative filtering (user ratings data) for a hybrid system that improves with usage.
- **Fuzzy title search** — Use edit distance (Levenshtein) or phonetic matching so misspelled titles still find the right movie.
- **Genre / decade filters** — Let users narrow recommendations by genre, release decade, or minimum rating.
- **User feedback** — Thumbs up/down on recommendations to fine-tune results over time.
- **Caching** — Cache TMDB API responses server-side to reduce API calls and improve poster load speed.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset by [The Movie Database (TMDB)](https://www.themoviedb.org/) via Kaggle
- Word2Vec algorithm — Mikolov et al., Google (2013): *"Efficient Estimation of Word Representations in Vector Space"*
- [Gensim](https://radimrehurek.com/gensim/) library by Radim Řehůřek
- Deployed on [Render](https://render.com)
