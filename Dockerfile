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

# Start the pipeline
CMD ["python", "main.py", "--backfill"]
