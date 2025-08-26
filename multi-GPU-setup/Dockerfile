# Multi-GPU PyTorch Docker Environment
# Based on NVIDIA CUDA 11.8 runtime with Python 3.9
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    curl \
    vim \
    htop \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python (remove existing first)
RUN rm -f /usr/bin/python /usr/bin/python3
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copy application files
COPY test_gpus.py .
COPY multi_gpu_example_fixed.py .
COPY gpu_monitor.py .

# Create a non-root user for better security
RUN useradd -m -s /bin/bash dockeruser && \
    chown -R dockeruser:dockeruser /workspace
USER dockeruser

# Set environment variables for the user
ENV PATH="/home/dockeruser/.local/bin:$PATH"

# Expose port for Jupyter (if needed)
EXPOSE 8888

# Default command
CMD ["python", "test_gpus.py"]
