# Build stage for AI Micro-Expression Analyzer API
FROM python:3.11-slim

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Set entry point
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
