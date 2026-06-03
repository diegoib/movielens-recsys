FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --group train --group data --group onnx --no-dev

COPY src/ ./src/

ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "python", "src/train.py"]
