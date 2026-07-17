FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the running app actually needs (notebooks/, docs/, .git are excluded via .dockerignore).
COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
