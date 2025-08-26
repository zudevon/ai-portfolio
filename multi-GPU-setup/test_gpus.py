import torch
import sys

def test_gpu_setup():
    """Test GPU setup and show information about available GPUs"""
    print("=== GPU Setup Test ===\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Show information about each GPU
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test basic tensor operations on each GPU
        print("\n=== Testing GPU Operations ===")
        for i in range(torch.cuda.device_count()):
            print(f"\nTesting GPU {i}:")
            torch.cuda.set_device(i)
            
            # Create a test tensor
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.mm(x, y)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            print(f"  Matrix multiplication (1000x1000): {elapsed_time:.2f} ms")
            print(f"  Result shape: {z.shape}")
            
            # Clear memory
            del x, y, z
            torch.cuda.empty_cache()
        
        # Test multi-GPU operations
        print("\n=== Testing Multi-GPU Operations ===")
        if torch.cuda.device_count() > 1:
            print("Testing DataParallel...")
            try:
                # Create a simple model
                model = torch.nn.Linear(100, 10)
                model = torch.nn.DataParallel(model)
                model = model.cuda()
                
                # Test forward pass
                x = torch.randn(32, 100).cuda()
                output = model(x)
                print(f"  DataParallel output shape: {output.shape}")
                
                del model, x, output
                torch.cuda.empty_cache()
                print("  DataParallel test passed!")
                
            except Exception as e:
                print(f"  DataParallel test failed: {e}")
        
        print("\n=== GPU Test Complete ===")
        
    else:
        print("CUDA is not available. Please check your PyTorch installation.")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = test_gpu_setup()
        if success:
            print("\n✅ GPU setup is working correctly!")
        else:
            print("\n❌ GPU setup has issues.")
    except Exception as e:
        print(f"\n❌ Error during GPU test: {e}")
        import traceback
        traceback.print_exc()
