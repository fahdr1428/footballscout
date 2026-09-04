# Container image for hosts other than Streamlit Community Cloud
# (Hugging Face Spaces, Render, Railway, Fly.io, Cloud Run, a VPS).
#
#   docker build -t footballscout .
#   docker run -p 8501:8501 footballscout
#
# The datasets are committed to the repository, so the image is self-contained:
# there is no build-time network fetch and no volume to mount.

FROM python:3.11-slim

# Streamlit writes its config and stats here; give it a real home directory.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    PORT=8501

WORKDIR /app

# Dependencies first, so edits to the app do not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','8501')}/_stcore/health\")"

# Shell form so $PORT expands: hosts that assign a port at runtime (Render,
# Railway, Fly, Cloud Run) set it, and it falls back to 8501 locally.
CMD streamlit run app.py \
        --server.port="${PORT}" \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
