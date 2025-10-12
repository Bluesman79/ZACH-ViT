# ZACH-ViT: A Zero-Token Adaptive Compact Hierarchical Vision Transformer

Official implementation of **ZACH-ViT**, a lightweight Vision Transformer for robust classification of lung ultrasound videos.  
Introduced in *Angelakis et al., 2025 (in review, THe Lancet Digital Health)*.

---

## Overview
**ZACH-ViT** redefines Vision Transformer design for small, heterogeneous medical datasets.

- No positional embeddings or class tokens  
- Dynamic adaptive residuals for stable feature learning  
- Global pooling for order-agnostic representations  
- **ShuffleStrides Data Augmentation (SSDA)**: structured permutation-based augmentation preserving clinical plausibility  

| Model               | Parameters (M) | Validation AUC | Test AUC | Sensitivity | Specificity |
| ------------------- | -------------- | -------------- | -------- | ----------- | ----------- |
| **ZACH-ViT (ours)** | **0.25**       | **0.80**       | **0.79** | 0.60        | 0.91        |
| Standard ViT        | 0.62           | 0.58           | 0.54     | 0.10        | 0.97        |
| ResNet50            | 23.8           | 0.65           | 0.54     | 0.05        | 0.94        |
| DenseNet121         | 7.17           | 0.64           | 0.53     | 0.05        | 0.95        |

---

## ⚙️ Usage
```bash
# Clone the repository
git clone https://github.com/Bluesman79/ZACH-ViT.git
cd ZACH-ViT

# Install dependencies
pip install -r requirements.txt

# Train ZACH-ViT
python training/train_tf.py --config configs/zachvit.yaml
