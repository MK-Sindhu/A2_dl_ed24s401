# A2_dl_ed24s401

# Deep Learning Assignment 2  

This repository contains two parts of the assignment on CNN‐based image classification using a subset of the iNaturalist dataset:

- **Part A**: Train a small CNN from scratch  
- **Part B**: Fine‑tune a pre‑trained ResNet50  

---

## 📁 Repository Structure

```
A2_dl_ed24s401/
├── Part_A/
│   ├── Part_A.ipynb
│   ├── train_partA_scratch.py
│   ├── eval_partA_scratch.py
│   └── best_model.pth
│
├── Part_B/
│   ├── Part_B.ipynb
│   ├── train_best_resnet50.py
│   ├── eval_partB_resnet.py
│   └── best_resnet50.pth
│
├── inaturalist_12K/        ← **YOU MUST DOWNLOAD & CREATE THIS**
│   ├── train/              ← 10 classes of training images
│   └── val/               ← 10 classes of test images
│
└── README.md               ← This file
```

---

## 📂 Dataset Setup

This repository does **not** include the iNaturalist images. Before running any code:

1. **Download** the iNaturalist 12K subset (train + test) from your course or Kaggle link.  
2. **In the project root**, create a directory named `inaturalist_12K` with two subfolders:

   ```
   A2_dl_ed24s401/
   ├── inaturalist_12K/
   │   ├── train/
   │   └── val/
   └── ...
   ```

3. **Place** all training images under `inaturalist_12K/train/` and all test images under `inaturalist_12K/val/`.

---

## ⚙️ Environment Setup

```bash
# 1. Clone this repository
git clone https://github.com/<your‑username>/A2_dl_ed24s401.git
cd A2_dl_ed24s401

# 2. (Optional) create & activate a virtualenv
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision numpy matplotlib tqdm wandb
```

---

## 📚 Part A: Train CNN From Scratch

### 1. Training

```bash
cd Part_A
python train_partA_scratch.py \
  --data_dir ../inaturalist_12K \
  --batch_size 32 \
  --epochs 10
```

- **`--data_dir`**: points to `inaturalist_12K` (scripts add `/train` or `/test`).  
- Uses **20%** of training set for validation.  
- Saves best checkpoint as `best_model.pth`.

### 2. Evaluation

```bash
python eval_partA_scratch.py \
  --data_dir ../inaturalist_12K \
  --batch_size 32
```

- Loads `best_model.pth`.  
- Evaluates on **test** split.  
- Prints **Test Accuracy** (≈ 35.65%).

---

## 📚 Part B: Fine‑Tune Pre‑trained ResNet50

### 1. Training

```bash
cd Part_B
python train_best_resnet50.py \
  --data_dir ../inaturalist_12K \
  --batch_size 64 \
  --epochs 6 \
  --lr 1e-4 \
  --augment true
```

- Applies ImageNet normalization & augmentation.  
- Splits **20%** for validation.  
- Saves best checkpoint as `best_resnet50.pth`.

### 2. Evaluation

```bash
python eval_partB_resnet.py \
  --data_dir ../inaturalist_12K \
  --batch_size 64
```

- Loads `best_resnet50.pth`.  
- Evaluates on **test** split.  
- Prints **Test Accuracy** (≈ 86.75%).

---

## 📈 Results Summary

| Model                          | Best Val Acc    | Test Acc  |
|--------------------------------|:---------------:|:---------:|
| **Scratch CNN (Part A)**       | 36.15% (epoch 7) | 35.65%    |
| **ResNet50 fine‑tuned (B)**    | 85.50% (epoch 4) | 86.75%    |

---

## 🔧 Final Hyperparameters

### Part A (Scratch CNN)
- **Filters**: 32  
- **Kernel**: 3×3  
- **Activation**: Mish  
- **Batch size**: 32  
- **Batchnorm**: No  
- **Dropout**: 0  
- **Hidden units**: 256  
- **Learning rate**: 1e‑3  
- **Epochs**: 10  

### Part B (ResNet50)
- **Fine‑tune strategy**: full_finetune (all layers)  
- **Batch size**: 64  
- **Augmentation**: True  
- **Learning rate**: 1e‑4  
- **Epochs**: 6  
- **Validation split**: 0.2  

---

## 📊 W&B Sweeps

- **Part A**: Random search over CNN hyperparameters  
- **Part B**: Bayesian optimization over ResNet50 fine‑tuning parameters  

🔗 View all runs & plots:  
https://wandb.ai/ed24s401-indian-institute-of-technology-madras/assignment_2_sweep

---

## 📝 Github repo link

🔗 https://github.com/MK-Sindhu/A2_dl_ed24s401

