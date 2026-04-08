## Segmentation and masking with UNET

* link




## UNET for masking and segmentation


```


# =========================
# SIMPLE U-NET SEGMENTATION (MNIST)
# =========================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# DATA
# =========================
transform = T.ToTensor()

train_ds = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# =========================
# CREATE SEGMENTATION TARGET
# =========================
# mask = 1 where pixel > threshold

def make_mask(x):
    return (x > 0.1).float()

# =========================
# SIMPLE U-NET
# =========================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # down
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # up
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )

        # output mask
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        m = self.mid(e2)

        u = self.up(m)
        cat = torch.cat([u, e1], dim=1)

        d = self.dec1(cat)

        return torch.sigmoid(self.out(d))  # mask [0,1]

# =========================
# MODEL
# =========================
model = SimpleUNet().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# =========================
# TRAIN
# =========================
EPOCHS = 3

for epoch in range(EPOCHS):
    for xb, _ in train_dl:
        xb = xb.to(device)

        mask = make_mask(xb)

        pred = model(xb)

        loss = loss_fn(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} loss: {loss.item():.4f}")

# =========================
# TEST + VISUALIZE
# =========================
model.eval()

xb, _ = next(iter(train_dl))
xb = xb.to(device)

with torch.no_grad():
    pred = model(xb)

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(9,3))

for i in range(3):
    # original
    plt.subplot(3,3, i+1)
    plt.imshow(xb[i,0].cpu(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    # ground truth
    plt.subplot(3,3, i+4)
    plt.imshow(make_mask(xb[i]).cpu()[0], cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    # prediction
    plt.subplot(3,3, i+7)
    plt.imshow(pred[i,0].cpu(), cmap='gray')
    plt.title("Pred")
    plt.axis('off')

plt.tight_layout()
plt.show()



```



## UNET for segmentation of objects DATA (Pascal VOC)

```


# =========================
# SIMPLE U-NET SEGMENTATION (Cars & Bikes)
# =========================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# DATA (Pascal VOC)
# =========================
transform = T.Compose([
    T.Resize((128,128)),
    T.ToTensor()
])

target_transform = T.Compose([
    T.Resize((128,128)),
    T.PILToTensor()
])

train_ds = torchvision.datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='train',
    download=True,
    transform=transform,
    target_transform=target_transform
)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

# =========================
# FILTER CLASSES (car=7, bike=2)
# =========================
def simplify_mask(mask):
    # VOC class IDs
    CAR = 7
    BIKE = 2

    new_mask = torch.zeros_like(mask)

    new_mask[mask == CAR] = 1
    new_mask[mask == BIKE] = 2

    return new_mask.long()

# =========================
# SIMPLE U-NET (3 classes)
# =========================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU())

        self.mid = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU())

        self.up = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64,32,3,padding=1), nn.ReLU())

        self.out = nn.Conv2d(32, 3, 1)  # 3 classes

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        m = self.mid(e2)

        u = self.up(m)
        cat = torch.cat([u, e1], dim=1)

        d = self.dec1(cat)

        return self.out(d)  # logits

# =========================
# MODEL
# =========================
model = SimpleUNet().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# =========================
# TRAIN (short demo)
# =========================
for epoch in range(2):
    for img, mask in train_dl:

        img = img.to(device)

        mask = simplify_mask(mask.squeeze(1)).to(device)

        pred = model(img)

        loss = loss_fn(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("epoch", epoch, "loss", loss.item())

# =========================
# VISUALIZE
# =========================
model.eval()

img, mask = next(iter(train_dl))
img = img.to(device)

with torch.no_grad():
    pred = model(img)
    pred = torch.argmax(pred, dim=1)

plt.figure(figsize=(9,3))

for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(img[i].permute(1,2,0).cpu())
    plt.title("Image")
    plt.axis('off')

    plt.subplot(3,3,i+4)
    plt.imshow(mask[i][0], cmap='jet')
    plt.title("GT")
    plt.axis('off')

    plt.subplot(3,3,i+7)
    plt.imshow(pred[i].cpu(), cmap='jet')
    plt.title("Pred")
    plt.axis('off')

plt.show()


```


