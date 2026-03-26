FROM python:3.11-slim

WORKDIR /app

# System deps for gensim (Cython extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"

# Copy app code
COPY . .

# Expose port
EXPOSE 5000

# Single worker keeps RAM under 512 MB (free-tier Render/Railway)
# --timeout 120 for the similarity computation on cold start
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--preload", "app:app"]
