# A2_dl_ed24s401

# Deep Learning Assignment 2  


This repository contains implementations for:

- **Part A**: Training a small CNN from scratch on a subset of the iNaturalist dataset  
- **Part B**: Fine‑tuning a pre‑trained ResNet50 on the same data  

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


README.md                      # This file
```

---

## 🎯 Dataset Preparation

1. Download the iNaturalist 12K subset.  
2. Create a folder next to this README:

   ```bash
   mkdir inaturalist_12K
   mv <your_downloaded_train> inaturalist_12K/train
   mv <your_downloaded_test>  inaturalist_12K/test
   ```

---

## ⚙️ Environment Setup

```bash
# Clone repo
git clone https://github.com/<your‑username>/A2_dl_ed24s401.git
cd A2_dl_ed24s401

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm wandb
```

---

## 📚 Part A: Train CNN From Scratch

### 1. Train

```bash
cd Part_A
python train_partA_scratch.py
```

- Reads data from `../inaturalist_12K/train`  
- Uses 20% of training data for validation  
- Trains for 10 epochs  
- Saves best model to `best_model.pth`  

### 2. Evaluate

```bash
python eval_partA_scratch.py
```

- Loads `best_model.pth`  
- Evaluates on `../inaturalist_12K/test`  
- Prints **Test Accuracy** (~ 35.65%)  

---

## 📚 Part B: Fine‑Tune Pre‑trained ResNet50

### 1. Train

```bash
cd Part_B
python train_best_resnet50.py
```

- Reads data from `../inaturalist_12K/train`  
- Applies ImageNet normalization & optional augmentation  
- Splits 20% for validation  
- Fine‑tunes for 6 epochs at LR = 1e‑4  
- Saves best model to `best_resnet50.pth`  

### 2. Evaluate

```bash
python eval_partB_resnet.py
```

- Loads `best_resnet50.pth`  
- Evaluates on `../inaturalist_12K/test`  
- Prints **Test Accuracy** (~ 86.75%)  

---

## 📈 Results Summary

| Model                        | Best Val Acc      | Test Acc  |
|------------------------------|:-----------------:|:---------:|
| Scratch CNN (Part A)         | 36.15% (epoch 7)  | 35.65%    |
| ResNet50 fine‑tuned (Part B) | 85.50% (epoch 4)  | 86.75%    |

---

## 🔧 Hyperparameters (“Sweet Spots”)

- **Part A (Scratch CNN)**  
  - Filters = 32, Kernel = 3×3, Activation = Mish  
  - Batch size = 32, No augmentation/batchnorm, Dropout = 0  
  - Hidden units = 256, LR = 1e‑3  

- **Part B (ResNet50)**  
  - Strategy = full_finetune (all layers trainable)  
  - Batch size = 64, Augment = True, LR = 1e‑4  
  - Epochs = 6, Val split = 0.2  

---

## 📊 W&B Sweeps

- **Part A**: Random search over CNN hyperparameters  
- **Part B**: Bayesian optimization over ResNet50 fine‑tuning settings  

View all runs and interactive plots at:  
https://wandb.ai/ed24s401-indian-institute-of-technology-madras/assignment_2_sweep

