FROM python:3.11-slim

# Install Tesseract (English) and clean apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Railway provides PORT environment variable automatically
ENV PORT=8080
CMD ["python", "main.py"]
