

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (Docker layer caching — if requirements don't
# change, this layer is cached and pip install doesn't re-run on every push)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create models directory (weights downloaded at runtime from HF Hub)
RUN mkdir -p models

# HF Spaces runs as UID 1000 — ensure write permissions
RUN chmod -R 777 /app

# Expose port 7860 — required by Hugging Face Spaces
EXPOSE 7860

# Run the Streamlit app
# --server.port 7860          : HF Spaces required port
# --server.address 0.0.0.0    : listen on all interfaces
# --server.headless true      : no browser auto-open
# --server.fileWatcherType none : disable file watcher (not needed in prod)
CMD ["streamlit", "run", "app/app.py",
     "--server.port", "7860",
     "--server.address", "0.0.0.0",
     "--server.headless", "true",
     "--server.fileWatcherType", "none"]