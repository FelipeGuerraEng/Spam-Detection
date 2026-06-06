FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY app ./app
RUN python -c "from app.huggingface_utils import HUGGINGFACE_MODEL_ID, load_huggingface_model; load_huggingface_model(HUGGINGFACE_MODEL_ID)"

COPY models ./models
COPY reports ./reports
COPY spam.csv .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
