#!/bin/bash
# Run Assignment-5 Docker container on GPU-1 only

set -e

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker daemon is not running. Start Docker first, then rerun this script."
    exit 1
fi

# Build the image
echo "Building Docker image..."
docker build -t assignment5-mlops .

# Run container with GPU-1 only and increased shared memory for data loaders
echo "Starting container with GPU-1..."
docker run --gpus '"device=1"' --rm -it \
    --shm-size=8g \
    --ipc=host \
    -v "$(pwd):/workspace" \
    -e CUDA_VISIBLE_DEVICES=0 \
    assignment5-mlops bash
