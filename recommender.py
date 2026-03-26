"""
CineMatch — Word2Vec Recommender Engine
========================================
NLP pipeline:
  1. Feature extraction  — overview, genres, keywords, cast, director
  2. Weighted tag corpus — overview sentences repeated 3× for importance
  3. Lemmatization       — WordNetLemmatizer (falls back to basic clean)
  4. Word2Vec training   — Skip-gram, 300-dim, window=5, trained on this corpus
  5. Movie embeddings    — mean-pooled Word2Vec vectors per movie
  6. Cosine similarity   — pairwise across all ~4800 movies

Why Word2Vec beats TF-IDF + LSA here:
  - Captures semantic similarity: "warrior" ~ "soldier", "space" ~ "galaxy"
  - Trained specifically on movie language — not generic Wikipedia text
  - Mean-pooled doc vectors are compact (~300 floats vs sparse 10k matrix)
  - No vocabulary size cap — every word contributes if it has been seen
"""

import ast
import os
import pickle
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── optional NLTK (graceful fallback) ────────────────────────────────────────
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    nltk.download('wordnet',   quiet=True)
    nltk.download('stopwords', quiet=True)
    _lem   = WordNetLemmatizer()
    _stops = set(stopwords.words('english'))
    USE_NLTK = True
except Exception:
    USE_NLTK = False
    _lem   = None
    _stops = set()


# ── text helpers ──────────────────────────────────────────────────────────────

def _lemmatize(word):
    return _lem.lemmatize(word) if USE_NLTK else word


def _tokenize(text):
    """Lowercase, strip punctuation, remove stopwords, lemmatize."""
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        _lemmatize(t)
        for t in text.split()
        if t not in _stops and len(t) > 2
    ]
    return tokens


def _clean_name(name):
    """'Sam Worthington' -> 'SamWorthington' (one unambiguous token)"""
    return name.replace(" ", "")


# ── JSON column parsers ───────────────────────────────────────────────────────

def _parse_list(obj):
    try:
        return [i["name"] for i in ast.literal_eval(obj)]
    except Exception:
        return []


def _top3(obj):
    try:
        return [i["name"] for i in ast.literal_eval(obj)][:3]
    except Exception:
        return []


def _director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i.get("job") == "Director":
                return [i["name"]]
    except Exception:
        pass
    return []


# ── core class ────────────────────────────────────────────────────────────────

class MovieRecommender:
    MODEL_PATH = "models/recommender.pkl"

    def __init__(self):
        self.df            = None
        self.similarity    = None
        self.movie_indices = None

    # ── BUILD ─────────────────────────────────────────────────────────────────

    def build(self, movies_csv, credits_csv):
        print("Loading CSVs...")
        movies  = pd.read_csv(movies_csv)
        credits = pd.read_csv(credits_csv)
        df = movies.merge(credits, on="title")

        keep = ["movie_id", "title", "overview", "genres", "keywords",
                "cast", "crew", "vote_average", "vote_count",
                "popularity", "release_date"]
        df = df[[c for c in keep if c in df.columns]].dropna(subset=["overview"])

        print("Extracting features...")
        df["genres_parsed"]   = df["genres"].apply(_parse_list)
        df["keywords_parsed"] = df["keywords"].apply(_parse_list)
        df["cast_parsed"]     = df["cast"].apply(_top3)
        df["director"]        = df["crew"].apply(_director)

        # Each movie becomes a list of tokens for Word2Vec training.
        # overview tokens * 3 so the model sees plot language more often.
        def movie_tokens(row):
            ov_tok    = _tokenize(row["overview"])
            genre_tok = [_clean_name(g).lower() for g in row["genres_parsed"]]
            kw_tok    = [_clean_name(k).lower() for k in row["keywords_parsed"]]
            cast_tok  = [_clean_name(c).lower() for c in row["cast_parsed"]]
            dir_tok   = [_clean_name(d).lower() for d in row["director"]]
            return (ov_tok * 3 + genre_tok * 2 + kw_tok * 2
                    + cast_tok + dir_tok * 2)

        df["tokens"] = df.apply(movie_tokens, axis=1)
        corpus = df["tokens"].tolist()

        # Train Word2Vec on the movie corpus
        print("Training Word2Vec (skip-gram, 300-dim, 15 epochs)...")
        w2v = Word2Vec(
            sentences   = corpus,
            vector_size = 300,
            window      = 5,
            min_count   = 2,
            workers     = 4,
            sg          = 1,      # skip-gram: better for rare words / small data
            epochs      = 15,
            seed        = 42,
        )

        # Mean-pool token vectors into one vector per movie
        print("Building movie embedding matrix...")

        def mean_vector(tokens):
            vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
            if not vecs:
                return np.zeros(w2v.vector_size)
            return np.mean(vecs, axis=0)

        embeddings = np.vstack([mean_vector(tok) for tok in corpus])
        embeddings = normalize(embeddings)          # L2-norm: cosine == dot

        print("Computing pairwise cosine similarity...")
        similarity = cosine_similarity(embeddings)

        self.df = df[["movie_id", "title", "overview", "genres_parsed",
                      "cast_parsed", "director", "vote_average",
                      "vote_count", "popularity", "release_date"]].reset_index(drop=True)
        self.similarity    = similarity
        self.movie_indices = pd.Series(self.df.index, index=self.df["title"])

        os.makedirs("models", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({"df": self.df, "similarity": self.similarity}, f)

        print("Done — {} movies indexed. Model saved to {}".format(
            len(self.df), self.MODEL_PATH))

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load(self):
        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.df            = data["df"]
        self.similarity    = data["similarity"]
        self.movie_indices = pd.Series(self.df.index, index=self.df["title"])

    def is_ready(self):
        return self.df is not None and self.similarity is not None

    # ── RECOMMEND ─────────────────────────────────────────────────────────────

    def recommend(self, title, n=10):
        if title not in self.movie_indices.index:
            return []
        idx    = self.movie_indices[title]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: n + 1]
        results = []
        for i, score in scores:
            row = self.df.iloc[i]
            results.append({
                "title":      row["title"],
                "movie_id":   int(row["movie_id"])           if pd.notna(row.get("movie_id"))      else None,
                "score":      round(float(score), 4),
                "genres":     row.get("genres_parsed", []),
                "overview":   str(row["overview"])           if pd.notna(row.get("overview"))      else "",
                "cast":       row.get("cast_parsed", []),
                "director":   row.get("director", []),
                "rating":     float(row["vote_average"])     if pd.notna(row.get("vote_average"))  else None,
                "vote_count": int(row["vote_count"])         if pd.notna(row.get("vote_count"))    else None,
                "year":       str(row["release_date"])[:4]   if pd.notna(row.get("release_date"))  else None,
            })
        return results

    def search(self, query, limit=12):
        q    = query.lower()
        mask = self.df["title"].str.lower().str.contains(q, na=False)
        return self.df[mask]["title"].head(limit).tolist()

    def all_titles(self):
        return self.df["title"].tolist()
