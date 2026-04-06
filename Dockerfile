FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Set environment variable to use only GPU-1 (mapped as cuda:0 inside container)
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["bash"]
