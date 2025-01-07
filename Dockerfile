FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV MODEL_PATH=/app/ml/v1/pkl
ENV DATASET_PATH=/app/data.csv
ENV VERSION=v1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY ml/v1/datasets/spam_detect_dataset.csv /app/data.csv

EXPOSE 8080

CMD ["python", "main.py"]
