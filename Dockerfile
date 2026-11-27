FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 PIP_NO_CACHE_DIR=1
RUN groupadd -r app && useradd -r -g app -u 1000 -m -d /app app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY download_model.py .
RUN python download_model.py
COPY --chown=app:app proxy.py context.py manage.py ./
RUN mkdir -p /data && chown app:app /data
USER app
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health', timeout=3)" || exit 1
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8081", "--loop", "uvloop", "--http", "httptools"]
