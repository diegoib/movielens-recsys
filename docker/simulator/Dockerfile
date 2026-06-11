FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --group simulator --no-dev

COPY src/ ./src/

ENV PYTHONPATH=/app
CMD ["uv", "run", "--no-sync", "python", "src/simulator/online_simulator.py"]
