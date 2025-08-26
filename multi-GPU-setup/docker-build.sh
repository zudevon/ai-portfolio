#!/bin/bash

# Multi-GPU Docker Build and Run Script
# This script builds and runs the Docker container with GPU access

set -e

echo "🚀 Multi-GPU Docker Environment Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Docker runtime not detected. Installing nvidia-docker2..."
    echo "Please run these commands as root:"
    echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-docker2"
    echo "  sudo systemctl restart docker"
    echo ""
    echo "After installation, run this script again."
    exit 1
fi

# Check if GPUs are available
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

echo ""
echo "📦 Building Docker image..."
docker-compose build

echo ""
echo "🚀 Starting multi-GPU environment..."
echo ""

# Run the container
docker-compose up -d multi-gpu-ml

echo ""
echo "✅ Container started successfully!"
echo ""
echo "📋 Available commands:"
echo "  • View logs: docker-compose logs -f multi-gpu-ml"
echo "  • Stop container: docker-compose down"
echo "  • Access container: docker exec -it multi-gpu-ml bash"
echo "  • Run GPU test: docker exec -it multi-gpu-ml python test_gpus.py"
echo "  • Run training: docker exec -it multi-gpu-ml python multi_gpu_example_fixed.py"
echo "  • Monitor GPUs: docker exec -it multi-gpu-ml python gpu_monitor.py"
echo ""
echo "🌐 Jupyter Lab available at: http://localhost:8889"
echo "📊 TensorBoard available at: http://localhost:6006"
echo ""
echo "🎯 Your multi-GPU Docker environment is ready!"
