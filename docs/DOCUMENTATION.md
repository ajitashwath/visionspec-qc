# VisionSpec QC - Visual Quality Control System
## Project Documentation

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Technical Architecture](#technical-architecture)
4. [Implementation Details](#implementation-details)
5. [Setup Instructions](#setup-instructions)
6. [Usage Guide](#usage-guide)
7. [Model Performance](#model-performance)
8. [Troubleshooting](#troubleshooting)
9. [Future Enhancements](#future-enhancements)

---

## Executive Summary

**VisionSpec QC** is an AI-powered visual quality control system designed for real-time PCB (Printed Circuit Board) defect detection on assembly lines. The system uses transfer learning with MobileNetV2 and Grad-CAM visualization to classify images as "Pass" or "Defect" with visual indication of defect locations.

### Key Features
- **Binary Classification**: Pass/Defect detection with >90% accuracy
- **Real-Time Processing**: Low-latency inference optimized for production
- **Visual Explainability**: Grad-CAM heatmaps show exactly where defects are located
- **Transfer Learning**: Pre-trained MobileNetV2 for high accuracy with limited data
- **Data Augmentation**: Robust to lighting variations and camera angles

---

## Project Overview

### Business Context
Manufacturing assembly lines for PCBs require 100% quality inspection for soldering defects. Manual inspection is:
- Time-consuming
- Inconsistent
- Prone to human error
- Expensive at scale

### Solution
VisionSpec QC provides automated, real-time visual inspection using deep learning to:
1. Capture images via high-speed camera
2. Classify as Pass/Defect in milliseconds
3. Highlight defect locations using Grad-CAM
4. Reduce false negatives and improve throughput

### Success Metrics
- **Accuracy**: >90% on validation set
- **Latency**: <100ms per image
- **Explainability**: Clear visual indicators of decision reasoning
- **Robustness**: Handles lighting and angle variations

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   VisionSpec QC System                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  Camera      │─────▶│  Preprocessing│               │
│  │  Input       │      │  (224x224)    │               │
│  └──────────────┘      └───────┬───────┘               │
│                                │                        │
│                                ▼                        │
│                   ┌────────────────────┐                │
│                   │  MobileNetV2       │                │
│                   │  (Transfer Learning)│               │
│                   └─────────┬──────────┘                │
│                             │                           │
│                  ┌──────────┴──────────┐                │
│                  │                     │                │
│                  ▼                     ▼                │
│          ┌──────────────┐      ┌─────────────┐         │
│          │ Classification│      │  Grad-CAM   │         │
│          │ (Pass/Defect) │      │  Heatmap    │         │
│          └──────────────┘      └─────────────┘         │
│                  │                     │                │
│                  └──────────┬──────────┘                │
│                             ▼                           │
│                   ┌──────────────────┐                  │
│                   │  Visual Output   │                  │
│                   │  + Confidence    │                  │
│                   └──────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | TensorFlow/Keras 2.12+ | Model training & inference |
| **Computer Vision** | OpenCV 4.x | Image processing & display |
| **Base Model** | MobileNetV2 | Pre-trained feature extraction |
| **Visualization** | Matplotlib | Training metrics & analysis |
| **Explainability** | Grad-CAM | Defect localization |

### Model Architecture

```
Input Image (224×224×3)
         ↓
┌──────────────────────────────────┐
│   MobileNetV2 Base (Frozen)      │
│   - 53 Convolutional Layers      │
│   - Pre-trained on ImageNet      │
│   - Depth-wise Separable Convs   │
└────────────┬─────────────────────┘
             ↓
    Global Average Pooling (1280)
             ↓
    Dense Layer (128, ReLU)
             ↓
    Dropout (0.5)
             ↓
    Output Dense (1, Sigmoid)
             ↓
    Prediction [0-1]
```

**Total Parameters**: 2,422,081
- Trainable: 164,097 (custom layers)
- Non-trainable: 2,257,984 (frozen MobileNetV2)

---

## Implementation Details

### Week 1: Data Pipeline & Preprocessing

**Objectives**:
- ✓ Load and organize image dataset
- ✓ Implement real-time data augmentation
- ✓ Standardize preprocessing

**Implementation** (`notebooks/data_prep.ipynb`):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize to [0,1]
    rotation_range=20,        # Random rotation ±20°
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    zoom_range=0.15,          # Random zoom
    horizontal_flip=True,     # Mirror flipping
    fill_mode='nearest'       # Fill strategy
)

val_datagen = ImageDataGenerator(rescale=1./255)
```

**Data Organization**:
```
data/processed/
├── train/
│   ├── pass/       (80% of Pass images)
│   └── defect/     (80% of Defect images)
└── validation/
    ├── pass/       (20% of Pass images)
    └── defect/     (20% of Defect images)
```

**Validation**: Visual inspection confirms augmented samples are realistic and diverse.

---

### Week 2: Transfer Learning & Training

**Objectives**:
- ✓ Load pre-trained MobileNetV2
- ✓ Add custom classification head
- ✓ Train with frozen base layers
- ✓ Monitor for overfitting

**Implementation** (`notebooks/training.ipynb`):

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Load pre-trained base
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze layers

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
```

**Training Configuration**:
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 20
- **Batch Size**: 32

**Performance Monitoring**:
- Loss curves track training vs validation loss
- Accuracy curves show model improvement
- Early stopping prevents overfitting

---

### Week 3: Grad-CAM Explainability

**Objectives**:
- ✓ Implement Grad-CAM algorithm
- ✓ Generate heatmaps for predictions
- ✓ Validate visual explanations

**Implementation** (`notebooks/gradcam_analysis.ipynb`):

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

**Grad-CAM Process**:
1. Forward pass to get feature maps and prediction
2. Compute gradient of prediction w.r.t. feature maps
3. Pool gradients to get importance weights
4. Weight feature maps and sum
5. Apply ReLU and normalize

**Visualization**:
```python
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return img, heatmap, superimposed
```

---

### Week 4: Real-Time Inference Engine

**Objectives**:
- ✓ Optimize model for fast inference
- ✓ Build live video stream processing
- ✓ Integrate Grad-CAM for defects
- ✓ Demonstrate low-latency performance

**Implementation** (`src/inference.py`):

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = 'models/mobile_net_v2_pcb.h5'
CONFIDENCE_THRESHOLD = 0.5

# Load model once
model = load_model(MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Preprocess
    input_frame = cv2.resize(frame, (224, 224))
    input_array = np.expand_dims(input_frame, axis=0) / 255.0
    
    # Predict
    prediction = model.predict(input_array, verbose=0)[0][0]
    
    # Apply Grad-CAM for defects
    if prediction > CONFIDENCE_THRESHOLD:
        label = "DEFECT"
        color = (0, 0, 255)  # Red
        heatmap = generate_gradcam(input_array, model, 'out_relu')
        frame = apply_gradcam_overlay(frame, heatmap)
    else:
        label = "PASS"
        color = (0, 255, 0)  # Green
    
    # Display
    text = f"{label}: {prediction:.2%}"
    cv2.putText(frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('VisionSpec QC', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

**Performance Optimizations**:
- Model loaded once (not per frame)
- Batch prediction disabled for single images
- OpenCV for fast image operations
- Conditional Grad-CAM (only for defects)

---

## Setup Instructions

### Prerequisites

- **Python**: 3.8 - 3.11
- **GPU**: Optional but recommended (CUDA-compatible)
- **Camera**: USB/integrated camera for live inference

### Step 1: Clone Repository

```bash
git clone https://github.com/ajitashwath/visionspec-qc.git
cd visionspec-qc
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install tensorflow==2.12.0 opencv-python matplotlib scikit-learn
```

**Note**: TensorFlow 2.11.0 is not compatible with Python 3.11. Use TensorFlow 2.12+ or Python 3.10.

### Step 4: Prepare Dataset

```bash
mkdir -p data/processed/{train,validation}/{pass,defect}
```

Place your images:
- `data/processed/train/pass/` - Good PCBs (80%)
- `data/processed/train/defect/` - Defective PCBs (80%)
- `data/processed/validation/pass/` - Good PCBs (20%)
- `data/processed/validation/defect/` - Defective PCBs (20%)

### Step 5: Train Model

```bash
jupyter notebook notebooks/training.ipynb
```

Run all cells to train the model. The trained model will be saved to `models/mobile_net_v2_pcb.h5`.

---

## Usage Guide

### Training from Scratch

1. **Prepare Data** (Week 1):
   ```bash
   jupyter notebook notebooks/data_prep.ipynb
   ```
   - Verify augmentation is working
   - Check data distribution

2. **Train Model** (Week 2):
   ```bash
   jupyter notebook notebooks/training.ipynb
   ```
   - Monitor loss curves
   - Validate no overfitting
   - Save best model

3. **Analyze with Grad-CAM** (Week 3):
   ```bash
   jupyter notebook notebooks/gradcam_analysis.ipynb
   ```
   - Test on validation images
   - Verify heatmaps highlight defects
   - Check misclassifications

### Real-Time Inference

```bash
python src/inference.py
```

**Controls**:
- Press `q` to quit
- Ensure camera is connected
- Model will show:
  - Green "PASS" for good PCBs
  - Red "DEFECT" with heatmap overlay for defects

### Batch Prediction

```python
from src.utils import preprocess_image
from tensorflow.keras.models import load_model

model = load_model('models/mobile_net_v2_pcb.h5')

# Single image
img = preprocess_image('path/to/image.jpg')
prediction = model.predict(img)[0][0]
print(f"Defect Probability: {prediction:.2%}")
```

---

## Model Performance

### Expected Metrics

Based on typical transfer learning results:

| Metric | Target | Typical Range |
|--------|--------|---------------|
| **Accuracy** | >90% | 92-96% |
| **Precision** | >85% | 88-94% |
| **Recall** | >85% | 87-93% |
| **F1 Score** | >85% | 88-93% |
| **Inference Time** | <100ms | 30-80ms (CPU) |

### Training Insights

**Good Training**:
- Loss decreases smoothly
- Val loss tracks train loss
- Accuracy improves steadily
- No divergence between train/val

**Overfitting Signs**:
- Val loss increases while train loss decreases
- Large gap between train/val accuracy
- **Solution**: Increase dropout, add more augmentation

**Underfitting Signs**:
- Both losses remain high
- Low accuracy on train set
- **Solution**: Train longer, unfreeze some base layers

### Grad-CAM Validation

**Good Heatmaps**:
- Focus on soldering joints
- Highlight actual defect areas
- Consistent across similar defects

**Poor Heatmaps**:
- Focus on background
- Random highlighting
- **Action**: Retrain with more diverse data

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install tensorflow==2.12.0
```

#### 2. Camera Not Opening

**Error**: `Error: Cannot access camera`

**Solution**:
- Check camera connection
- Try different camera index:
  ```python
  cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
  ```
- Grant camera permissions (macOS/Linux)

#### 3. Out of Memory (GPU)

**Error**: `ResourceExhaustedError: OOM`

**Solution**:
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### 4. Low Accuracy

**Possible Causes**:
- Insufficient training data
- Poor data quality
- Class imbalance

**Solutions**:
1. Add more augmentation
2. Collect more images
3. Use class weights:
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       'balanced',
       classes=np.unique(train_labels),
       y=train_labels
   )
   ```

#### 5. Grad-CAM Not Working

**Error**: `Layer 'out_relu' not found`

**Solution**:
```python
# Find correct layer name
for layer in model.layers:
    print(layer.name)

# Use last convolutional layer before pooling
LAST_CONV_LAYER = 'Conv_1_bn'  # Adjust based on output
```

---

## Future Enhancements

### Short-Term (1-3 months)

1. **Multi-Class Classification**
   - Classify defect types: cold joint, bridging, insufficient solder
   - Requires labeled dataset with defect categories

2. **Model Optimization**
   - TensorFlow Lite conversion for edge devices
   - Quantization for 4x size reduction
   - ONNX export for cross-platform deployment

3. **API Development**
   - REST API for remote inference
   - Batch processing endpoint
   - WebSocket for real-time streaming

### Mid-Term (3-6 months)

4. **Advanced Augmentation**
   - GAN-based synthetic defect generation
   - Mix-up and cut-mix strategies
   - Test-time augmentation

5. **Ensemble Models**
   - Combine MobileNetV2 + EfficientNet
   - Voting or weighted averaging
   - Improve robustness

6. **Active Learning Pipeline**
   - Flag uncertain predictions
   - Human-in-the-loop labeling
   - Continuous model improvement

### Long-Term (6-12 months)

7. **Multi-Camera System**
   - Multiple viewing angles
   - 3D defect reconstruction
   - Synchronized capture

8. **Production Integration**
   - PLC/SCADA integration
   - Automatic rejection mechanism
   - Database logging

9. **Edge Deployment**
   - NVIDIA Jetson deployment
   - Raspberry Pi optimization
   - On-device inference

---

## Project Structure

```
visionspec-qc/
├── data/
│   └── processed/
│       ├── train/
│       │   ├── pass/
│       │   └── defect/
│       └── validation/
│           ├── pass/
│           └── defect/
├── models/
│   └── mobile_net_v2_pcb.h5
├── notebooks/
│   ├── data_prep.ipynb
│   ├── training.ipynb
│   └── gradcam_analysis.ipynb
├── src/
│   ├── inference.py
│   └── utils.py
├── QUICKSTART.md
├── README.md
└── requirements.txt
```

---

## Dependencies

```txt
tensorflow>=2.12.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
numpy>=1.23.0,<2.0.0
```

---

## Contact & Support

For issues, questions, or contributions:
- Create an issue in the repository
- Email: support@visionspec-qc.com

---

## Acknowledgments

- MobileNetV2 architecture from Google Research
- Grad-CAM implementation inspired by original paper (Selvaraju et al., 2017)
- TensorFlow/Keras community for excellent documentation