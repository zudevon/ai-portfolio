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

def train_single_gpu(model, train_loader, criterion, optimizer, device, num_epochs=5):
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
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        total_loss += epoch_loss
        print(f"  Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Single GPU training completed in {training_time:.2f} seconds")
    return total_loss / num_epochs

def train_multi_gpu(model, train_loader, criterion, optimizer, num_epochs=5):
    """Train model using DataParallel on multiple GPUs"""
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        total_loss += epoch_loss
        print(f"  Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Multi-GPU training completed in {training_time:.2f} seconds")
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
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    X, y = create_dummy_data(num_samples=50000, input_size=784, num_classes=10)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    # Test single GPU training
    print("\n--- Single GPU Training ---")
    model_single = SimpleNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_single.parameters(), lr=0.001)
    
    single_loss = train_single_gpu(model_single, train_loader, criterion, optimizer, 'cuda')
    
    # Test multi-GPU training
    print("\n--- Multi-GPU Training ---")
    model_multi = SimpleNet().cuda()
    optimizer_multi = optim.Adam(model_multi.parameters(), lr=0.001)
    
    multi_loss = train_multi_gpu(model_multi, train_loader, criterion, optimizer_multi)
    
    # Compare results
    print("\n--- Results Comparison ---")
    print(f"Single GPU final loss: {single_loss:.4f}")
    print(f"Multi-GPU final loss: {multi_loss:.4f}")
    
    # Clean up
    del model_single, model_multi
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        benchmark_gpus()
        print("\n✅ Multi-GPU benchmark completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
