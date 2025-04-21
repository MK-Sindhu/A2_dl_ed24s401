# train_partA_scratch.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ─── 1) Settings ───────────────────────────────────────────────
DATA_ROOT   = './inaturalist_12K'               # must contain 'train/' and 'test/' subfolders
TRAIN_DIR   = os.path.join(DATA_ROOT, 'train')
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 10
VAL_FRAC    = 0.2
NUM_CLASSES = 10
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── 2) Transforms ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ─── 3) Stratified split ───────────────────────────────────────
def stratified_split(dataset, val_fraction):
    labels = np.array([y for _, y in dataset.samples])
    train_idxs, val_idxs = [], []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0].tolist()
        random.shuffle(idxs)
        cut = int((1 - val_fraction) * len(idxs))
        train_idxs += idxs[:cut]
        val_idxs   += idxs[cut:]
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)

# ─── 4) SmallCNN Definition ────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, in_ch, filt_list, kernel_size, activation,
                 use_bn, dropout_p, dense_units, num_classes, input_size):
        super().__init__()
        layers = []
        for i, f in enumerate(filt_list):
            layers.append(nn.Conv2d(in_ch if i == 0 else filt_list[i-1],
                                     f, kernel_size, padding=kernel_size//2))
            if use_bn: layers.append(nn.BatchNorm2d(f))
            layers.append(getattr(nn, activation)())
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        final_spatial = input_size // (2**len(filt_list))
        feat_dim = filt_list[-1] * final_spatial**2

        clf = [nn.Flatten(),
               nn.Linear(feat_dim, dense_units),
               getattr(nn, activation)()]
        if dropout_p > 0: clf.append(nn.Dropout(dropout_p))
        clf.append(nn.Linear(dense_units, num_classes))
        self.classifier = nn.Sequential(*clf)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ─── 5) DataLoaders ─────────────────────────────────────────────
full_ds       = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_ds, val_ds = stratified_split(full_ds, VAL_FRAC)
train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── 6) Build Model ────────────────────────────────────────────
# Best hyperparameters from Part A
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
else:  # "half"
    filt_list = [max(1, filters//(2**i)) for i in range(5)]

model = SmallCNN(3, filt_list, kernel_size, activation,
                 use_batchnorm, dropout_p, hidden_units,
                 NUM_CLASSES, IMG_SIZE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

# ─── 7) Train Loop ─────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    tr_loss, tr_corr, tr_tot = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
        preds    = out.argmax(1)
        tr_corr += (preds == y).sum().item()
        tr_tot  += y.size(0)
    tr_loss /= tr_tot
    tr_acc   = tr_corr / tr_tot

    # Validate
    model.eval()
    val_loss, val_corr, val_tot = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out   = model(x)
            loss  = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds     = out.argmax(1)
            val_corr += (preds == y).sum().item()
            val_tot  += y.size(0)
    val_loss /= val_tot
    val_acc   = val_corr / val_tot

    print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"→ New best model saved (val_acc={val_acc:.4f})")

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
