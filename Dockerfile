FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for native packages (hdbscan, numpy, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pulse listens here unless the platform injects its own PORT.
ENV PORT=8080
EXPOSE 8080

# Serve the Pulse web UI. The pipeline itself is triggered from the UI (or
# from the CLI: `python -m src.main --backfill`), so the container's job is
# to stay up and listen.
#
# Shell form is required: exec form does not expand $PORT. `python -m`
# avoids depending on the console script being on PATH.
CMD ["sh", "-c", "python -m uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
