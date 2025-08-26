@echo off
echo 🚀 Multi-GPU Docker Environment Setup
echo ======================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo 🔍 Checking GPU availability...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

echo.
echo 📦 Building Docker image...
docker-compose build

echo.
echo 🚀 Starting multi-GPU environment...
echo.

REM Run the container
docker-compose up -d multi-gpu-ml

echo.
echo ✅ Container started successfully!
echo.
echo 📋 Available commands:
echo   • View logs: docker-compose logs -f multi-gpu-ml
echo   • Stop container: docker-compose down
echo   • Access container: docker exec -it multi-gpu-ml bash
echo   • Run GPU test: docker exec -it multi-gpu-ml python test_gpus.py
echo   • Run training: docker exec -it multi-gpu-ml python multi_gpu_example_fixed.py
echo   • Monitor GPUs: docker exec -it multi-gpu-ml python gpu_monitor.py
echo.
echo 🌐 Jupyter Lab available at: http://localhost:8889
echo 📊 TensorBoard available at: http://localhost:6006
echo.
echo 🎯 Your multi-GPU Docker environment is ready!
echo.
pause
