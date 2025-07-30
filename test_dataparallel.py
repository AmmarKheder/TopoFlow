import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    if torch.cuda.device_count() > 1:
        print("\nTesting DataParallel...")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    # Test forward pass
    x = torch.randn(32, 10).cuda()
    try:
        y = model(x)
        print("DataParallel test successful!")
        print(f"Output shape: {y.shape}")
    except Exception as e:
        print(f"DataParallel test failed: {e}")
else:
    print("No GPUs available for testing")
