FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --group serving --no-dev

COPY src/ ./src/

ENV PYTHONPATH=/app
# Shell form so ${PORT:-8000} expands at runtime; Cloud Run injects PORT=8080
CMD uv run --no-sync uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}
