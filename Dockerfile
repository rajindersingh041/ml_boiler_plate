FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN /uv sync --frozen --no-dev
COPY src/ src/
COPY artifacts/production/ artifacts/production/
CMD [".venv/bin/uvicorn", "ml_boilerplate.api:app", "--host", "0.0.0.0", "--port", "8000"]
