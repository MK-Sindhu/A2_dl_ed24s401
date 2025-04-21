# eval_partB_resnet.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# ─── 1) Settings ─────────────────────────────────────────────────
DATA_ROOT   = "./inaturalist_12K"          # same root as training
TEST_DIR    = os.path.join(DATA_ROOT, "test")
MODEL_PATH  = "best_resnet50.pth"
BATCH_SIZE  = 64
IMAGE_SIZE  = 224
NUM_CLASSES = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─── 2) Transforms & DataLoader ───────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_ds     = datasets.ImageFolder(TEST_DIR, transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── 3) Rebuild ResNet50 & Load Weights ───────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc    = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ─── 4) Evaluate on Test Set ──────────────────────────────────────
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

print(f"🎯 Test Accuracy: {100 * correct/total:.2f}%")
