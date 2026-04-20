FROM python:3.11-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libreoffice \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p pdf_search

EXPOSE 10000
RUN mkdir -p pdf_search
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "main:app"]
