## Art Style Transfer

* Gatys Neural Transfer
* TorchPainterStyleTransfer notebook (https://arxiv.org/pdf/1902.11108.pdf)


```


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Fake target feature (what we want to match) ---
target_feature = torch.randn(1, 10)

# --- Loss layer ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)  # compute loss
        return x                                # pass through unchanged

# --- Simple "CNN" ---
model = nn.Sequential(
    nn.Linear(5, 10),        # acts like a conv layer
    nn.ReLU(),
    ContentLoss(target_feature),  # <-- loss inserted here
    nn.Linear(10, 3)
)

# --- Input (this is what we optimize in style transfer) ---
x = torch.randn(1, 5, requires_grad=True)

# --- Forward pass ---
output = model(x)

# --- Collect loss from inside the network ---
loss = model[2].loss   # ContentLoss layer

# --- Backprop ---
loss.backward()




```
