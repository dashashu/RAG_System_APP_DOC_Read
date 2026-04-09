FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/chroma_data /data \
    && chown -R appuser:appuser /app /data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data /data

USER appuser

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    CHROMA_DIR=/app/chroma_data

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
