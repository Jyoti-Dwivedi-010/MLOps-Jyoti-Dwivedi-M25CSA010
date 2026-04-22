FROM python:3.10-slim

# Setting the working directory
WORKDIR /app

# Installing system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY translate.py .
COPY input.txt .
COPY reference.txt .
COPY evaluate.py .

# Create output directory
RUN mkdir -p /app/output

# Set entry point
CMD ["python", "translate.py"]
