#!/usr/bin/env python3
"""
Docker GPU Test Script
This script verifies that GPUs are accessible within the Docker container
"""

import os
import sys
import torch

def print_system_info():
    """Print system and environment information"""
    print("=== Docker GPU Test ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check environment variables
    print(f"\nEnvironment Variables:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  NVIDIA_DRIVER_CAPABILITIES: {os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'Not set')}")

def test_gpu_access():
    """Test GPU access and functionality"""
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available in Docker container!")
        print("This usually means:")
        print("  1. NVIDIA Docker runtime is not properly configured")
        print("  2. GPU drivers are not accessible from container")
        print("  3. Docker run command missing --gpus all flag")
        return False
    
    print(f"\n✅ CUDA is available!")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Test each GPU
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"\nGPU {i}: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test basic operations
        try:
            torch.cuda.set_device(i)
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"  ✅ Basic operations: PASSED")
            print(f"  Matrix multiplication result shape: {z.shape}")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Basic operations: FAILED - {e}")
            return False
    
    return True

def test_multi_gpu():
    """Test multi-GPU functionality"""
    if torch.cuda.device_count() < 2:
        print("\n⚠️  Only one GPU available, skipping multi-GPU test")
        return True
    
    print(f"\n=== Multi-GPU Test ===")
    
    try:
        # Test DataParallel
        model = torch.nn.Linear(100, 10)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        
        # Test forward pass
        x = torch.randn(32, 100).cuda()
        output = model(x)
        print(f"✅ DataParallel test: PASSED")
        print(f"  Output shape: {output.shape}")
        
        # Clean up
        del model, x, output
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test: FAILED - {e}")
        return False

def main():
    """Main test function"""
    try:
        print_system_info()
        
        if not test_gpu_access():
            print("\n❌ GPU access test failed!")
            print("\nTroubleshooting steps:")
            print("1. Ensure Docker Desktop is running")
            print("2. Verify NVIDIA Docker runtime is available")
            print("3. Check that GPUs are visible from host (nvidia-smi)")
            print("4. Rebuild container with: docker-compose build --no-cache")
            sys.exit(1)
        
        if not test_multi_gpu():
            print("\n⚠️  Multi-GPU test failed, but single GPU is working")
        else:
            print("\n✅ Multi-GPU test passed!")
        
        print("\n🎉 Docker GPU test completed successfully!")
        print("\nYour Docker container has full access to:")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  • {gpu_name}")
        
        print(f"\nTotal GPU memory available: {sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1024**3:.1f} GB")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
