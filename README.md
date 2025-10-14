# 🧩 ZACH-ViT: Zero-Token Adaptive Compact Hierarchical Vision Transformer with ShuffleStrides Data Augmentation (SSDA)

Official implementation of **ZACH-ViT**, a lightweight Vision Transformer for robust and explainable classification of lung ultrasound videos, and the **ShuffleStrides Data Augmentation (SSDA)** algorithm.  

Introduced in *Angelakis et al., 2025 (in review, The Lancet Digital Health)*.

---

## 📘 Overview

**ZACH-ViT** redefines Vision Transformer design for small, heterogeneous medical datasets.

- ❌ **No positional embeddings or class tokens** — zero-token paradigm for order-agnostic feature extraction  
- ⚙️ **Adaptive hierarchical residuals** for stable feature learning  
- 🌍 **Global pooling** for invariant image-level representations  
- 🔄 **ShuffleStrides Data Augmentation (SSDA)** — permutation-based semi-supervised augmentation preserving clinical plausibility  

---

## 🧠 Full Pipeline

This repository provides a fully reproducible pipeline for **preprocessing**, **training**, and **evaluation**, available as both **Jupyter notebooks** and **pure Python scripts**:

1. **ROI extraction** from raw TALOS DICOM ultrasound recordings  
2. **VIS (Video Image Sequence)** creation per patient, concatenating frame strides from all probe positions  
3. **ShuffleStrides semi-supervised data augmentation (0-SSDA)** for robust domain generalization  
4. **ShuffleStrides semi-supervised data augmentation (SSDA_p)** for permutation-based learning enhancement  
5. **ZACH-ViT** training, validation, and testing with automatic time and metric reporting    

---

## 📂 Data Directory Structure

The `../Data` directory evolves from raw patient data to fully structured training datasets.

### 🧩 Before Preprocessing
```bash
../Data/
├── TALOS100/
└── TALOS122/
```

**Description:**
- Each folder contains the raw ultrasound recordings (`.dcm` format) for one patient across the four transducer positions
- Data is stored in DICOM format, which is standard for medical imaging

### 🔄 After Preprocessing
```bash
../Data/
├── 0_SSDA/             # Dataset with all 4! stride permutations (first SSDA regime)
├── 2_3_SSDA/           # Second-level SSDA with partial stride reordering
├── imgs/               # Auto-saved training and validation plots (timestamped)
├── Processed_ROI/      # Extracted pleural ROI frames per position
├── TALOS100/           # Original raw DICOMs (kept for reference)
├── TALOS122/           # Original raw DICOMs (kept for reference)
├── VIS/                # Generated VIS images per patient (concatenated stride representation)
├── train/
│   ├── 0/              # Non-CPE
│   └── 1/              # CPE
├── val/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
```

### 🧠 Notes

- **VIS** images represent one patient by vertically stacking the four position-specific stride sequences.
- **SSDA** folders contain automatically generated semi-supervised augmentations.
- **train**, **val**, and **test** directories follow the standard Keras `ImageDataGenerator` convention with subfolders `0` and `1` for binary classes.
- All training curves from the ZACH-ViT notebook are automatically saved in `../Data/imgs/` with a date-time prefix (e.g., `ZACH_ViT_training_20251014_183502.png`).
---

## 🚀 Usage

```bash
# Clone the repository and Install Dependencies
git clone https://github.com/Bluesman79/ZACH-ViT.git
cd ZACH-ViT

# Install dependencies
pip install -r requirements.txt
```

## 📓 Using Jupyter Notebooks
1. Run Preprocessing
   Open and run the notebook: `Preprocessing_ROI_VIS_0_SSDA_SSDA_p`.

   This will:
   *  Extract and crop the DICOM ROIs
   *  Generate VIS images
   *  Create 0-SSDA and SSDA_p datasets
   *  
2. Train and evaluate ZACH-ViT
   Open and run the notebook: `ZACH-ViT`.

   This will:
   * Train the model
   * Report training/inference times
   * Save learning curves automatically in `../Data/imgs/`

## 💻 Using Terminal (Pure Python Scripts)
```bash
# Run preprocessing (example)
python preprocessing/preprocess_pipeline.py

# Train and evaluate ZACH-ViT
python training/zach_vit_train.py
```

Both scripts mirror the logic of the notebooks and save identical output structures.

## 🔁 Data Flow Overview
```bash
TALOS DICOM
   │
   ▼
ROI Extraction
   │
   ▼
VIS Image Generation
   │
   ▼
ShuffleStrides Data Augmentation (SSDA)
   │
   ▼
Train / Val / Test Sets
   │
   ▼
ZACH-ViT Training and Evaluation
```

## 🧾 Citation
If you use this work, please cite:

```bibtex
@article{angelakis2025zachvit,
  author    = {Angelakis, A. and et al.},
  title     = {ZACH-ViT: Zero-Token Adaptive Compact Hierarchical Vision Transformer with ShuffleStrides Data Augmentation (SSDA)},
  journal   = {The Lancet Digital Health},
  year      = {2025},
  note      = {In review}
}
```
