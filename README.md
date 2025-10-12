# ZACH-ViT: A Zero-Token Vision Transformer with ShuffleStrides Data Augmentation (SSDA)

Official implementation of **ZACH-ViT**, a lightweight Vision Transformer for robust classification of lung ultrasound videos.  
Introduced in *Angelakis et al., 2025 (in review, The Lancet Digital Health)*.

---

## 📘 Overview

**ZACH-ViT** redefines Vision Transformer design for small, heterogeneous medical datasets.

- ❌ No positional embeddings or class tokens  
- ⚙️ Dynamic adaptive residuals for stable feature learning  
- 🌍 Global pooling for order-agnostic representations  
- 🔄 **ShuffleStrides Data Augmentation (SSDA):** permutation-based semi-supervised augmentation preserving clinical plausibility  

---

## 🧠 Pipeline

This repository provides a full reproducible preprocessing and training pipeline:

1. **ROI extraction** from TALOS DICOM ultrasound recordings  
2. **VIS (Video Image Sequence)** image generation  
3. **ShuffleStrides semi-supervised data augmentation (SSDA)**  
4. **ZACH-ViT** model training and evaluation  

Each numbered Jupyter notebook (`01_`–`05_`) corresponds to a stage of the pipeline.



## Usage
```bash
# Clone the repository
git clone https://github.com/Bluesman79/ZACH-ViT.git
cd ZACH-ViT

# Install dependencies
pip install -r requirements.txt

# Train ZACH-ViT
python training/train_tf.py --config configs/zachvit.yaml
