FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN /uv sync --frozen --no-dev
COPY src/ src/
# Production model is mounted at runtime (-v ./artifacts/production:/app/artifacts/production), not baked in.
RUN mkdir -p artifacts/production
CMD [".venv/bin/uvicorn", "ml_boilerplate.api:app", "--host", "0.0.0.0", "--port", "8000"]
