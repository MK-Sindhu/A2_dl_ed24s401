# train_partB_resnet.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset

# ─── Reproducibility ─────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─── 1) Hyperparameters & Paths ─────────────────────────────────────
DATA_ROOT    = "./inaturalist_12K"          # must contain "train/" and "test/"
TRAIN_DIR    = os.path.join(DATA_ROOT, "train")
MODEL_PATH   = "best_resnet50.pth"
BATCH_SIZE   = 64
LR           = 1e-4
EPOCHS       = 6
VAL_FRAC     = 0.2
NUM_CLASSES  = 10
IMAGE_SIZE   = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Transforms ───────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def make_transforms(augment: bool):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(ops)

# ─── 3) Stratified split helper ─────────────────────────────────────
def stratified_split(dataset, val_fraction):
    labels = np.array([label for _, label in dataset.samples])
    train_idxs, val_idxs = [], []
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0].tolist()
        random.shuffle(idxs)
        cut = int((1 - val_fraction) * len(idxs))
        train_idxs += idxs[:cut]
        val_idxs   += idxs[cut:]
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)

# ─── 4) Build & Adapt ResNet50 ──────────────────────────────────────
def build_resnet(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ─── 5) Single‐epoch train/val loop ─────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if train_mode:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss/total, correct/total

# ─── 6) Prepare DataLoaders ─────────────────────────────────────────
full_ds     = datasets.ImageFolder(TRAIN_DIR, transform=make_transforms(True))
train_ds, val_ds = stratified_split(full_ds, VAL_FRAC)
# no-augment for validation
val_ds.dataset.transform = make_transforms(False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── 7) Instantiate Model, Optimizer, Loss ─────────────────────────
model     = build_resnet(NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

# ─── 8) Training Loop ──────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train_mode=True)
    val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, train_mode=False)

    print(f"Epoch {epoch}/{EPOCHS} | "
          f"train_loss: {tr_loss:.4f}, train_acc: {tr_acc:.4f} | "
          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"→ New best model saved (val_acc={val_acc:.4f})")

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
