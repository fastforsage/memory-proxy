FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY proxy.py .
ENV DB_PATH=/data/memory.db
RUN mkdir -p /data
EXPOSE 8081
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8081"]
