FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py .
COPY utils/ ./utils/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 5000

CMD ["python", "app.py"]
