# VisionSpec QC - QUICKSTART

## 1. SETUP (5 minutes)

```bash
# Install dependencies
pip install tensorflow opencv-python matplotlib scikit-learn

# Create folder structure
mkdir -p data/processed/{train,validation}/{pass,defect}
mkdir -p notebooks models src
```

## 2. PREPARE DATA

Put your PCB images in:
- `data/processed/train/pass/` - Good PCBs
- `data/processed/train/defect/` - Defective PCBs
- `data/processed/validation/pass/` - Good PCBs (validation)
- `data/processed/validation/defect/` - Defective PCBs (validation)

Split: 80% train, 20% validation

## 3. RUN NOTEBOOKS (In Jupyter)

**Week 1:** `notebooks/01_data_prep.ipynb` - Verify augmentation
**Week 2:** `notebooks/02_training.ipynb` - Train model
**Week 3:** `notebooks/03_gradcam_analysis.ipynb` - Test explainability

## 4. RUN LIVE INFERENCE

```bash
python src/inference.py
```

Press 'q' to quit.

## THAT'S IT!

Model saves to: `models/mobile_net_v2_pcb.h5`