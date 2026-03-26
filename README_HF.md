---
title: CineMatch Movie Recommender
emoji: 🎬
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
---

# CineMatch — Word2Vec Movie Recommender

Deployed via Docker on Hugging Face Spaces (free tier).

## Usage
1. Pre-train the model locally: `python train.py`
2. Commit `models/recommender.pkl` to the repo (or use Git LFS for large files)
3. Push to Hugging Face — the Docker build handles the rest

The app runs on port 7860 by default on HF Spaces (set `PORT=7860` env var).
