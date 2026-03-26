"""
CineMatch — Movie Recommender API + Server
Run:  python app.py
"""

import os
from flask import Flask, render_template, jsonify, request, abort
from recommender import MovieRecommender

app = Flask(__name__)
rec = MovieRecommender()

# ── Auto-load model on startup ────────────────────────────────────────────────
if os.path.exists(rec.MODEL_PATH):
    rec.load()
    print("✅ Model loaded from disk.")
else:
    print("⚠️  No model found. POST /api/build to train (requires CSV files).")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title', '').strip()
    n = min(int(request.args.get('n', 10)), 20)
    if not title:
        return jsonify({'error': 'title parameter required'}), 400
    if not rec.is_ready():
        return jsonify({'error': 'Model not loaded yet'}), 503
    results = rec.recommend(title, n)
    if not results:
        return jsonify({'error': f'Movie "{title}" not found in dataset'}), 404
    return jsonify({'query': title, 'results': results})


@app.route('/api/search', methods=['GET'])
def search():
    q = request.args.get('q', '').strip()
    if not q or len(q) < 2:
        return jsonify([])
    if not rec.is_ready():
        return jsonify([])
    return jsonify(rec.search(q, limit=12))


@app.route('/api/status', methods=['GET'])
def status():
    ready = rec.is_ready()
    count = len(rec.df) if ready else 0
    return jsonify({'ready': ready, 'movie_count': count})


@app.route('/api/build', methods=['POST'])
def build():
    """
    Trigger model training.
    Expects JSON: {"movies_csv": "path/to/tmdb_5000_movies.csv",
                   "credits_csv": "path/to/tmdb_5000_credits.csv"}
    Or place files in the project root and call with empty body.
    """
    data = request.get_json(silent=True) or {}
    movies_csv  = data.get('movies_csv', 'tmdb_5000_movies.csv')
    credits_csv = data.get('credits_csv', 'tmdb_5000_credits.csv')

    if not os.path.exists(movies_csv):
        return jsonify({'error': f'File not found: {movies_csv}'}), 400
    if not os.path.exists(credits_csv):
        return jsonify({'error': f'File not found: {credits_csv}'}), 400

    try:
        rec.build(movies_csv, credits_csv)
        return jsonify({'status': 'ok', 'movies_indexed': len(rec.df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
