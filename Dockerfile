# 1. Use the official lightweight Python 3.13 base image
FROM python:3.13-slim

# 2. Copy uv binary from the official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set working directory inside the container
WORKDIR /app

# 4. Install dependencies from lock file (no dev deps in production image)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 5. Copy the rest of the project
COPY . .

# 6. Install the project itself
RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Run the FastAPI app using uvicorn
CMD [".venv/bin/python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
