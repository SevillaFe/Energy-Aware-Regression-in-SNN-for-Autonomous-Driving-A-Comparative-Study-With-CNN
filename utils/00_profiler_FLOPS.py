# ---- setup ----
import torch
import torch.nn as nn

# 1) Define your CNN PilotNet *exactly* as in the paper
#    (use your current implementation)
class PilotNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*1*18, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 10), nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)
    
class LaksNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(3840, 120),
            nn.ReLU(),
            nn.Linear(120, 20),
            nn.ReLU(),
            nn.Linear(20, 1)  # Output is a single steering angle value
        )
    def forward(self, x):
        return self.net(x)
    

class MiniNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
    # Conv Block 1: 66×200 → 31×98
    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2),           # → 16×15×48

    # Conv Block 2: 15×48 → 6×22
    nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2),           # → 32×3×11

    # Conv Block 3: 3×11 → 1×9
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),

    # Collapse to a 1×1 spatial map
    nn.AdaptiveAvgPool2d((1, 1)),  
    nn.Flatten(),              # → (batch_size, 64)

    # Tiny regression head
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)           # steering angle
)
    def forward(self, x):
        return self.net(x)

# 2) SNN wrapper that reuses the SAME conv/linear weights and runs S steps
#    (Replace ReLU with your snnTorch layers inside your real model; for FLOP counting,
#     what matters is that conv/linear math repeats S times.)
class TimeExpandWrapper(nn.Module):
    def __init__(self, backbone, S: int):
        super().__init__()
        self.backbone = backbone
        self.S = S
    def forward(self, x):
        # emulate S time steps by feeding the same frame S times;
        # for FLOPs, this repeats conv/linear math S times.
        out = None
        for _ in range(self.S):
            out = self.backbone(x)
        return out

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ---- fvcore FLOPs ----
from fvcore.nn import FlopCountAnalysis


device = "cpu"  # or "mps"
cnn = MiniNetCNN().to(device).eval()  # my architecture as defined

# Parameter counts (identical across CNN and SNN wrappers)
params = count_params(cnn)
print(f"Parameters (CNN & SNN): {params:,}")
sum(p.numel() for p in cnn.parameters() if p.requires_grad)

def flops(model, x):
    model.eval()
    with torch.no_grad():
        return FlopCountAnalysis(model, x).total()

# 1 step (baseline)
x1 = torch.randn(1, 1, 66, 200, device=device)
F1 = flops(cnn, x1)

# S = 25 and S = 50 treated as batch
x25 = torch.randn(25, 1, 66, 200, device=device)
x50 = torch.randn(50, 1, 66, 200, device=device)
F25 = flops(cnn, x25)
F50 = flops(cnn, x50)

print(f"FLOPs CNN (1 step): {F1:,.0f}")
print(f"FLOPs SNN-equivalent (S=25 via batch): {F25:,.0f}  (~{F25/F1:.1f}× CNN)")
print(f"FLOPs SNN-equivalent (S=50 via batch): {F50:,.0f}  (~{F50/F1:.1f}× CNN)")
print(f"Ratio F50/F25 = {F50/F25:.3f}  (expected ≈ 2.000)")
