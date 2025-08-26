# 🐳 Multi-GPU Docker Environment

This Docker setup provides a portable, reproducible multi-GPU environment that can access your local RTX 3060 and Tesla A2 GPUs while maintaining the same functionality as your local setup.

## 🎯 **What This Docker Setup Provides**

- ✅ **Portable Environment**: Same setup can run on any machine with Docker and NVIDIA GPUs
- ✅ **GPU Access**: Full access to your local RTX 3060 and Tesla A2 GPUs
- ✅ **Reproducible**: Exact same environment every time
- ✅ **Easy Sharing**: Share your environment with others
- ✅ **Isolation**: Clean environment separate from your system
- ✅ **Scalability**: Easy to deploy on multiple machines

## 🔧 **Prerequisites**

### **1. Docker Installation**
- **Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Install Docker Engine and Docker Compose
- **macOS**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

### **2. NVIDIA Docker Support**
- **Windows**: Docker Desktop with WSL2 backend (automatic)
- **Linux**: Install nvidia-docker2 runtime
- **macOS**: Limited GPU support (consider remote development)

### **3. System Requirements**
- NVIDIA GPU drivers installed
- CUDA 11.8 compatible drivers
- At least 8GB RAM available for Docker

## 🚀 **Quick Start**

### **Option 1: Automated Setup (Recommended)**

#### **Windows:**
```bash
# Double-click this file:
docker-build.bat
```

#### **Linux/macOS:**
```bash
# Make executable and run:
chmod +x docker-build.sh
./docker-build.sh
```

### **Option 2: Manual Setup**

```bash
# 1. Build the Docker image
docker-compose build

# 2. Start the container
docker-compose up -d multi-gpu-ml

# 3. Check container status
docker-compose ps
```

## 📁 **Docker Files Structure**

```
tech-portfolio/
├── Dockerfile                    # Multi-GPU PyTorch environment
├── docker-compose.yml            # Container orchestration
├── .dockerignore                 # Build optimization
├── docker-build.sh               # Linux/macOS build script
├── docker-build.bat              # Windows build script
├── DOCKER_README.md              # This documentation
├── requirements.txt              # Python dependencies
├── test_gpus.py                  # GPU verification script
├── multi_gpu_example_fixed.py    # Multi-GPU training example
└── gpu_monitor.py                # GPU monitoring tool
```

## 🔍 **Verifying GPU Access**

### **1. Check Container Status**
```bash
docker-compose ps
```

### **2. Test GPU Access**
```bash
# Run GPU test inside container
docker exec -it multi-gpu-ml python test_gpus.py
```

### **3. Check GPU Utilization**
```bash
# Monitor GPUs from host
nvidia-smi

# Monitor GPUs from container
docker exec -it multi-gpu-ml python gpu_monitor.py
```

## 🎮 **Using the Docker Environment**

### **Interactive Shell Access**
```bash
# Access container shell
docker exec -it multi-gpu-ml bash

# Inside container, you can run:
python test_gpus.py                    # Test GPUs
python multi_gpu_example_fixed.py      # Run training
python gpu_monitor.py                  # Monitor GPUs
```

### **Running Scripts Directly**
```bash
# Run GPU test
docker exec -it multi-gpu-ml python test_gpus.py

# Run training benchmark
docker exec -it multi-gpu-ml python multi_gpu_example_fixed.py

# Monitor GPU usage
docker exec -it multi-gpu-ml python gpu_monitor.py
```

### **Jupyter Lab Access**
```bash
# Start Jupyter service
docker-compose up -d jupyter

# Access at: http://localhost:8889
```

## 📊 **Performance Comparison**

| Environment | Setup Time | GPU Access | Portability | Isolation |
|-------------|------------|------------|-------------|-----------|
| **Local venv** | ✅ Fast | ✅ Full | ❌ None | ❌ None |
| **Docker** | ⚠️ Medium | ✅ Full | ✅ High | ✅ Full |

## 🔧 **Docker Configuration Details**

### **Base Image**
- **nvidia/cuda:11.8.0-runtime-ubuntu20.04**
- Optimized for CUDA 11.8 compatibility
- Matches your local PyTorch CUDA version

### **GPU Access**
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  - CUDA_VISIBLE_DEVICES=0,1
```

### **Volume Mounts**
- **Local workspace**: `.:/workspace` (for development)
- **Data directory**: `./data:/workspace/data` (for datasets)
- **Cache**: `~/.cache:/home/dockeruser/.cache` (for persistence)

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. GPU Not Accessible**
```bash
# Check NVIDIA Docker runtime
docker info | grep nvidia

# Verify GPU visibility
nvidia-smi

# Check container logs
docker-compose logs multi-gpu-ml
```

#### **2. Permission Issues**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### **3. Memory Issues**
```bash
# Check Docker memory limits
docker stats

# Increase Docker memory in Docker Desktop settings
# Recommended: 8GB+ for ML workloads
```

#### **4. Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8888
netstat -tulpn | grep :6006

# Modify ports in docker-compose.yml if needed
```

### **Useful Commands**

```bash
# View container logs
docker-compose logs -f multi-gpu-ml

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Clean up Docker system
docker system prune -a

# Check GPU usage from host
nvidia-smi -l 1
```

## 🔄 **Development Workflow**

### **1. Local Development**
```bash
# Edit files locally (they're mounted in container)
# Changes are immediately available in container
```

### **2. Testing in Container**
```bash
# Test your changes
docker exec -it multi-gpu-ml python your_script.py
```

### **3. Iterative Development**
```bash
# No need to rebuild for code changes
# Rebuild only for dependency changes
docker-compose build
```

## 📈 **Scaling and Deployment**

### **Multi-Machine Deployment**
```bash
# Copy Docker files to other machines
# Ensure NVIDIA Docker runtime is installed
# Run docker-compose up -d
```

### **Production Considerations**
```bash
# Use specific GPU devices
environment:
  - CUDA_VISIBLE_DEVICES=0,1

# Set resource limits
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 8G
```

## 🎓 **Advanced Usage**

### **Custom Docker Images**
```dockerfile
# Extend the base image
FROM multi-gpu-ml:latest

# Add your custom packages
RUN pip install your-package

# Copy your code
COPY your_code/ /workspace/your_code/
```

### **Multi-Service Setup**
```yaml
# Add more services to docker-compose.yml
services:
  training:
    build: .
    runtime: nvidia
    # ... configuration
  
  inference:
    build: .
    runtime: nvidia
    # ... configuration
```

## 📞 **Support**

### **Getting Help**
1. Check the troubleshooting section above
2. Verify Docker and NVIDIA Docker runtime are working
3. Check container logs: `docker-compose logs`
4. Ensure your GPUs are accessible from the host

### **Useful Resources**
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)

---

## 🎉 **Success Summary**

Your Docker multi-GPU environment provides:
- ✅ **Full GPU access** to RTX 3060 and Tesla A2
- ✅ **Portable environment** that can run anywhere
- ✅ **Reproducible setup** for consistent results
- ✅ **Easy sharing** with team members
- ✅ **Professional deployment** capabilities

**🚀 Your multi-GPU environment is now containerized and ready for production use!**
