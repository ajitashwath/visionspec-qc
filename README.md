# VisionSpec QC - Visual Quality Control System

AI-powered real-time PCB defect detection using Transfer Learning and Grad-CAM visualization.

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

### 2. Organize Your Data
```bash
mkdir -p data/processed/{train,validation}/{pass,defect}
```

Put your PCB images in:
- `data/processed/train/pass/` - Good PCBs (80%)
- `data/processed/train/defect/` - Defective PCBs (80%)
- `data/processed/validation/pass/` - Good PCBs (20%)
- `data/processed/validation/defect/` - Defective PCBs (20%)

### 3. Train Model
```bash
jupyter notebook notebooks/training.ipynb
# Run all cells
```

### 4. Run Live Inference
```bash
python src/inference.py
# Press 'q' to quit
```

## 📊 What You Get

- ✅ **Real-time defect detection** with webcam
- ✅ **Visual explanations** using Grad-CAM heatmaps
- ✅ **90%+ accuracy** with transfer learning
- ✅ **Low latency** (<100ms per image)

## 🎯 Features

1. **Binary Classification**: Pass/Defect
2. **Transfer Learning**: MobileNetV2 pre-trained model
3. **Data Augmentation**: Rotation, zoom, flip for robustness
4. **Grad-CAM**: Shows WHERE the defect is located
5. **Real-time Processing**: Live camera feed inference

## 📁 Project Structure

```
visionspec-qc/
├── data/processed/         # Your PCB images
├── models/                 # Trained model (auto-generated)
├── notebooks/              # Jupyter notebooks
│   ├── data_prep.ipynb    # Week 1: Data pipeline
│   ├── training.ipynb     # Week 2: Model training
│   └── gradcam_analysis.ipynb  # Week 3: Explainability
└── src/
    ├── inference.py       # Week 4: Live inference
    └── utils.py           # Helper functions
```

## 🔧 Troubleshooting

### Camera not working?
```python
# Try different camera index in inference.py
cap = cv2.VideoCapture(1)  # Change 0 to 1, 2, etc.
```

### Out of memory?
```python
# Reduce batch size in training.ipynb
BATCH_SIZE = 16  # Default is 32
```

### Low accuracy?
- Add more training images
- Increase augmentation
- Train for more epochs

## 📖 Full Documentation

See `PROJECT_DOCUMENTATION.md` for complete details on:
- Architecture
- Implementation details
- Performance metrics
- Advanced troubleshooting

## 🎓 Learning Path (4 Weeks)

| Week | Focus | Notebook |
|------|-------|----------|
| 1 | Data Pipeline & Augmentation | `data_prep.ipynb` |
| 2 | Transfer Learning Training | `training.ipynb` |
| 3 | Grad-CAM Visualization | `gradcam_analysis.ipynb` |
| 4 | Real-time Inference | `inference.py` |

## ⚙️ System Requirements

- **Python**: 3.8-3.11
- **RAM**: 8GB minimum
- **GPU**: Optional (speeds up training)
- **Camera**: Any USB/integrated camera


## 🤝 Support
For issues or questions, see detailed troubleshooting in `PROJECT_DOCUMENTATION.md`.

---

**Built with**: TensorFlow • OpenCV • MobileNetV2 • Grad-CAM