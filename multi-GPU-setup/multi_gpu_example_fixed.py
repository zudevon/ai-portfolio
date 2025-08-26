import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=784, hidden_size=512, output_size=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_dummy_data(num_samples=10000, input_size=784, num_classes=10):
    """Create dummy data for training"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

def train_single_gpu(model, train_loader, criterion, optimizer, device, num_epochs=3):
    """Train model on a single GPU"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        total_loss += epoch_loss
        print(f"  Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Single GPU training completed in {training_time:.2f} seconds")
    return total_loss / num_epochs

def train_multi_gpu_balanced(model, train_loader, criterion, optimizer, num_epochs=3):
    """Train model using balanced multi-GPU approach"""
    print(f"  Using GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # Create separate models for each GPU to avoid memory imbalance issues
    models = []
    optimizers = []
    
    for i in range(torch.cuda.device_count()):
        gpu_model = SimpleNet().cuda(i)
        gpu_optimizer = optim.Adam(gpu_model.parameters(), lr=0.001)
        models.append(gpu_model)
        optimizers.append(gpu_optimizer)
    
    # Split data across GPUs
    total_loss = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Split batch across GPUs
            batch_size = data.size(0)
            gpu_batch_size = batch_size // len(models)
            
            total_batch_loss = 0
            
            for gpu_idx, (gpu_model, gpu_optimizer) in enumerate(zip(models, optimizers)):
                if gpu_idx == len(models) - 1:  # Last GPU gets remaining samples
                    gpu_data = data[gpu_idx * gpu_batch_size:].cuda(gpu_idx)
                    gpu_target = target[gpu_idx * gpu_batch_size:].cuda(gpu_idx)
                else:
                    gpu_data = data[gpu_idx * gpu_batch_size:(gpu_idx + 1) * gpu_batch_size].cuda(gpu_idx)
                    gpu_target = target[gpu_idx * gpu_batch_size:(gpu_idx + 1) * gpu_batch_size].cuda(gpu_idx)
                
                gpu_optimizer.zero_grad()
                output = gpu_model(gpu_data)
                loss = criterion(output, gpu_target)
                loss.backward()
                gpu_optimizer.step()
                
                total_batch_loss += loss.item()
            
            epoch_loss += total_batch_loss / len(models)
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Avg Loss: {total_batch_loss/len(models):.4f}")
        
        total_loss += epoch_loss
        print(f"  Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Multi-GPU training completed in {training_time:.2f} seconds")
    
    # Clean up
    for model in models:
        del model
    for optimizer in optimizers:
        del optimizer
    
    return total_loss / num_epochs

def benchmark_gpus():
    """Benchmark different GPU configurations"""
    print("=== Multi-GPU Training Benchmark ===\n")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    X, y = create_dummy_data(num_samples=20000, input_size=784, num_classes=10)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Test single GPU training (RTX 3060)
    print("\n--- Single GPU Training (RTX 3060) ---")
    torch.cuda.set_device(0)
    model_single = SimpleNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_single.parameters(), lr=0.001)
    
    single_loss = train_single_gpu(model_single, train_loader, criterion, optimizer, 'cuda:0')
    
    # Clean up
    del model_single, optimizer
    torch.cuda.empty_cache()
    
    # Test multi-GPU training
    print("\n--- Multi-GPU Training (Balanced) ---")
    multi_loss = train_multi_gpu_balanced(None, train_loader, criterion, None)
    
    # Compare results
    print("\n--- Results Comparison ---")
    print(f"Single GPU (RTX 3060) final loss: {single_loss:.4f}")
    print(f"Multi-GPU final loss: {multi_loss:.4f}")
    
    print("\n=== GPU Benchmark Complete ===")

if __name__ == "__main__":
    try:
        benchmark_gpus()
        print("\n✅ Multi-GPU benchmark completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
