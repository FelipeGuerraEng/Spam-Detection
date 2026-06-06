FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN python -c "from app.huggingface_utils import HUGGINGFACE_MODEL_ID, load_huggingface_model; load_huggingface_model(HUGGINGFACE_MODEL_ID)"

COPY models ./models
COPY reports ./reports
COPY spam.csv .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
