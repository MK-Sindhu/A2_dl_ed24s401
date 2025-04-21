# eval_partA_scratch.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─── 1) Settings ───────────────────────────────────────────────
DATA_ROOT   = './inaturalist_12K'               # same root as training
TEST_DIR    = os.path.join(DATA_ROOT, 'val')
BATCH_SIZE  = 32
IMG_SIZE    = 224
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── 2) Transforms & Loader ────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
test_ds     = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── 3) SmallCNN Definition ────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, in_ch, filt_list, kernel_size, activation,
                 use_bn, dropout_p, dense_units, num_classes, input_size):
        super().__init__()
        layers=[]
        for i,f in enumerate(filt_list):
            layers.append(nn.Conv2d(in_ch if i==0 else filt_list[i-1],
                                     f, kernel_size, padding=kernel_size//2))
            if use_bn: layers.append(nn.BatchNorm2d(f))
            layers.append(getattr(nn, activation)())
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        final_spatial = input_size // (2**len(filt_list))
        feat_dim = filt_list[-1] * final_spatial**2
        clf=[nn.Flatten(),
             nn.Linear(feat_dim, 256),
             getattr(nn, activation)()]
        clf.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*clf)
    def forward(self, x):
        return self.classifier(self.features(x))

# ─── 4) Instantiate & Load Weights ─────────────────────────────
filters        = 32
filter_organ   = "constant"
kernel_size    = 3
activation     = "Mish"
use_batchnorm  = False
dropout_p      = 0.0
hidden_units   = 256

if filter_organ == "constant":
    filt_list = [filters]*5
elif filter_organ == "double":
    filt_list = [filters*(2**i) for i in range(5)]
else:
    filt_list = [max(1, filters//(2**i)) for i in range(5)]

model = SmallCNN(3, filt_list, kernel_size, activation,
                 use_batchnorm, dropout_p, hidden_units,
                 NUM_CLASSES, IMG_SIZE).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ─── 5) Evaluate ────────────────────────────────────────────────
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        preds  = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

print(f"🎯 Test Accuracy: {100 * correct/total:.2f}%")
