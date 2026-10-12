# Multi-stage / Production Dockerfile for Docpilot
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies (build tools, libraries for Pillow, ReportLab, and OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . /app/

# Make scripts executable
RUN chmod +x /app/build.sh /app/start.sh

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Expose web port
EXPOSE 8000

# Run both the Telegram bot daemon and Gunicorn web server
CMD ["bash", "start.sh"]
