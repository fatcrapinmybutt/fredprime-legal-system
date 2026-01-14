FROM python:3.12-slim

# Create app directory
WORKDIR /app

# Install system deps for common packages (minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose application port (FastAPI/uvicorn)
EXPOSE 8000

# Run the health app (which exposes /health, /ready, /metrics)
CMD ["uvicorn", "src.health:app", "--host", "0.0.0.0", "--port", "8000"]
