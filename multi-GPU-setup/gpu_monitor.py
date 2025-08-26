import torch
import time
import psutil
import os

def get_gpu_info():
    """Get detailed information about all GPUs"""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print("=== GPU Information ===")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print(f"  Max Threads per Block: {props.max_threads_per_block}")
        print(f"  Max Shared Memory per Block: {props.max_shared_memory_per_block / 1024:.1f} KB")

def monitor_gpu_usage(duration=60, interval=2):
    """Monitor GPU usage in real-time"""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print(f"\n=== GPU Usage Monitor (Monitoring for {duration} seconds) ===")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            current_time = time.time() - start_time
            print(f"GPU Monitor - Elapsed: {current_time:.1f}s / {duration}s")
            print("=" * 50)
            
            # System info
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            print(f"CPU Usage: {cpu_percent:.1f}%")
            print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f} GB / {memory.total / 1024**3:.1f} GB)")
            print()
            
            # GPU info
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                
                # Memory info
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                # Utilization (approximate)
                utilization = (allocated / total) * 100
                
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {allocated:.2f} GB / {total:.1f} GB ({utilization:.1f}%)")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Free: {total - allocated:.2f} GB")
                print()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    print("GPU monitoring completed.")

def test_gpu_performance():
    """Test basic GPU performance metrics"""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print("\n=== GPU Performance Test ===")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nTesting GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.set_device(i)
        
        # Test matrix multiplication performance
        sizes = [1000, 2000, 4000]
        for size in sizes:
            # Warm up
            x = torch.randn(size, size).cuda()
            y = torch.randn(size, size).cuda()
            _ = torch.mm(x, y)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.mm(x, y)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            # Calculate FLOPS
            flops = 2 * size**3  # 2*n^3 for matrix multiplication
            gflops = (flops / elapsed_time) / 1e6
            
            print(f"  {size}x{size} matrix: {elapsed_time:.2f} ms, {gflops:.1f} GFLOPS")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()

def main():
    """Main function"""
    print("🚀 GPU Monitor and Performance Tool")
    print("=" * 40)
    
    # Show GPU information
    get_gpu_info()
    
    # Test performance
    test_gpu_performance()
    
    # Ask user if they want to monitor
    print("\n" + "=" * 40)
    response = input("Would you like to start real-time GPU monitoring? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        try:
            duration = int(input("Enter monitoring duration in seconds (default 60): ") or "60")
            monitor_gpu_usage(duration)
        except ValueError:
            print("Invalid duration, using default 60 seconds")
            monitor_gpu_usage()
    else:
        print("Monitoring skipped. Use 'python gpu_monitor.py' to run again.")

if __name__ == "__main__":
    main()
