FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements.txt changes)
COPY risk-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY risk-engine/ .

# Create data directory for persistent files
RUN mkdir -p /data

# Railway injects $PORT at runtime; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
