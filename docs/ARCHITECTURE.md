# VisionSpec QC - Technical Architecture Guide

## System Overview
VisionSpec QC is built on a modular architecture separating data processing, model training, and inference into distinct components.

## Architecture Layers

### 1. Data Layer
**Purpose**: Image acquisition, preprocessing, and augmentation

**Components**:
- `ImageDataGenerator`: Real-time augmentation
- Data loaders: Batch generation
- Preprocessing pipeline: Resize, normalize

**Flow**:
```
Raw Images → Resize (224x224) → Normalize [0,1] → Augment → Batch
```

### 2. Model Layer
**Purpose**: Feature extraction and classification

**Components**:
```
Input (224×224×3)
    ↓
MobileNetV2 Base (Frozen)
├── Conv Blocks (53 layers)
├── Depthwise Separable Convolutions
└── Feature Maps (7×7×1280)
    ↓
Global Average Pooling → (1280)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(1, Sigmoid) → [0,1]
```

**Key Design Decisions**:

1. **MobileNetV2 Base**
   - **Why**: Optimized for speed (depthwise convolutions)
   - **Alternative**: ResNet50 (higher accuracy, slower)
   - **Trade-off**: Balanced accuracy vs latency

2. **Frozen Base Layers**
   - **Why**: Preserve ImageNet features
   - **Benefits**: Faster training, prevents overfitting
   - **When to unfreeze**: If accuracy <85% after 20 epochs

3. **Global Average Pooling**
   - **Why**: Reduces parameters vs Flatten
   - **Benefits**: Regularization, spatial invariance
   - **Output**: 1280 features → 128 → 1

4. **Dropout 0.5**
   - **Why**: Prevent overfitting on small datasets
   - **Effect**: 50% neurons disabled during training
   - **Trade-off**: Slower convergence vs better generalization

### 3. Explainability Layer
**Purpose**: Visual interpretation using Grad-CAM

**Algorithm**:
```python
# Pseudo-code
1. Forward pass → Get feature maps (7×7×1280)
2. Get prediction score
3. Compute gradients: ∂(score)/∂(feature_maps)
4. Global average pool gradients → weights (1280)
5. Weighted sum: Σ(weights × feature_maps)
6. ReLU (remove negative)
7. Upsample to input size (224×224)
8. Overlay on original image
```

**Key Formula**:
```
Grad-CAM = ReLU(Σ_k α_k * A_k)

Where:
- A_k = k-th feature map
- α_k = 1/Z * Σ_i Σ_j ∂y/∂A_k_ij (global average pooled gradient)
- y = class score
```

### 4. Inference Layer
**Purpose**: Real-time prediction with low latency

**Pipeline**:
```
Camera Frame
    ↓
Resize (224×224)
    ↓
Normalize [0,1]
    ↓
Model.predict()
    ↓
Threshold (0.5)
    ↓
If Defect: Generate Grad-CAM
    ↓
Display: Label + Confidence + Heatmap
```

**Optimizations**:
1. Model loaded once (not per frame)
2. No batch dimension overhead
3. Conditional Grad-CAM (only for defects)
4. OpenCV for fast image ops

## Data Flow Diagram

```
┌─────────────┐
│  Raw Images │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Data Augmentation   │
│ - Rotation ±20°     │
│ - Shift 20%         │
│ - Zoom 15%          │
│ - Horizontal Flip   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Preprocessing       │
│ - Resize 224×224    │
│ - Normalize [0,1]   │
└──────┬──────────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│ Train Set   │   │ Val Set     │
│ (80%)       │   │ (20%)       │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────┐
│      MobileNetV2 Base       │
│   (Pre-trained, Frozen)     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Classification Head       │
│   - GAP                     │
│   - Dense(128)              │
│   - Dropout(0.5)            │
│   - Dense(1, sigmoid)       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│      Binary Output          │
│   0 = Pass, 1 = Defect      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│      Grad-CAM               │
│   (If Defect Detected)      │
└─────────────────────────────┘
```

## Component Details

### Data Augmentation Strategy

**Rationale**: PCB defects vary due to:
- Camera angle changes
- Lighting variations
- Component rotation
- Position shifts

**Augmentation Parameters**:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `rotation_range` | 20° | PCBs may be slightly rotated |
| `width_shift` | 20% | Components may be off-center |
| `height_shift` | 20% | Vertical misalignment |
| `zoom_range` | 15% | Camera distance varies |
| `horizontal_flip` | True | Mirror symmetry |
| `fill_mode` | 'nearest' | Preserve edge pixels |

**Effect**: 
- Original dataset: N images
- Augmented dataset: ~10-20× N variations
- No disk space increase (real-time)

### Transfer Learning Strategy

**Phase 1: Feature Extraction** (Week 2)
```python
base_model.trainable = False  # Freeze all layers
# Train only custom head (164k params)
epochs = 20
```

**Phase 2: Fine-Tuning** (Optional, if accuracy <85%)
```python
base_model.trainable = True
# Unfreeze top layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
# Train with lower learning rate
optimizer = Adam(lr=0.0001)  # 10x smaller
epochs = 10
```

### Grad-CAM Implementation Details

**Target Layer Selection**:
- Layer name: `'out_relu'` (last ReLU before pooling)
- Why: Highest-level features before classification
- Shape: (7, 7, 1280)

**Gradient Computation**:
```python
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]  # Binary classification score

grads = tape.gradient(loss, conv_outputs)
```

**Weighting & Aggregation**:
```python
# Global average pooling of gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (1280,)

# Weighted combination
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (7,7,1)
```

**Post-processing**:
```python
# Remove negative values
heatmap = tf.maximum(heatmap, 0)

# Normalize to [0,1]
heatmap = heatmap / tf.math.reduce_max(heatmap)

# Upsample to input size
heatmap_resized = cv2.resize(heatmap, (224, 224))

# Apply colormap (JET: blue → red)
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), 
                                     cv2.COLORMAP_JET)

# Overlay with alpha blending
output = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
```

## Performance Analysis

### Inference Latency Breakdown

| Operation | CPU Time (ms) | GPU Time (ms) | % of Total |
|-----------|---------------|---------------|------------|
| Image Resize | 2-3 | 2-3 | 5% |
| Normalization | 1 | 1 | 2% |
| Model Forward Pass | 40-60 | 10-15 | 70-80% |
| Grad-CAM (if needed) | 20-30 | 5-8 | 15-20% |
| Display | 3-5 | 3-5 | 5-8% |
| **Total** | **66-99ms** | **21-34ms** | **100%** |

**Conclusion**: GPU provides 3-4× speedup

### Memory Usage

| Component | RAM (MB) | VRAM (MB) |
|-----------|----------|-----------|
| Model Weights | 35 | 35 |
| Input Batch (32) | 50 | 50 |
| Activations | 150 | 150 |
| Gradients (Grad-CAM) | 100 | 100 |
| **Total** | **335MB** | **335MB** |

**Minimum Requirements**: 2GB RAM, 1GB VRAM

## Scalability Considerations

### Horizontal Scaling
```
Camera 1 → Inference Server 1 ─┐
Camera 2 → Inference Server 2 ─┤
Camera 3 → Inference Server 3 ─┼→ Load Balancer → Database
Camera 4 → Inference Server 4 ─┤
...                            ─┘
```

### Optimization Techniques

1. **Model Compression**
   ```python
   # TensorFlow Lite conversion
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   # Size reduction: 35MB → 9MB (4x)
   ```

2. **Quantization**
   ```python
   # INT8 quantization
   converter.target_spec.supported_types = [tf.int8]
   # Speed improvement: 2-3x
   # Accuracy drop: <1%
   ```

3. **Batch Inference**
   ```python
   # Process multiple images together
   batch = np.array([img1, img2, img3, img4])  # (4, 224, 224, 3)
   predictions = model.predict(batch)  # (4, 1)
   # Throughput: 4x higher
   ```

## Security & Reliability

### Error Handling

```python
# Robust inference pipeline
try:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture frame")
        continue
    
    prediction = model.predict(input_array, verbose=0)[0][0]
    
except Exception as e:
    logging.exception(f"Inference failed: {e}")
    # Fallback: Mark as "uncertain"
    prediction = 0.5
```

### Monitoring

**Key Metrics**:
- Frames per second (FPS)
- Average latency
- Defect detection rate
- False positive rate
- Model confidence distribution

**Logging**:
```python
import logging

logging.basicConfig(
    filename='inference.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Prediction: {prediction:.4f}, Label: {label}")
```

## Deployment Architectures

### Option 1: Edge Deployment
```
Camera → Edge Device (Jetson/RPi) → Model Inference → Display/Alert
```
**Pros**: Low latency, no network dependency
**Cons**: Limited compute power

### Option 2: Cloud Deployment
```
Camera → API Client → Cloud Server → Model Inference → Response
```
**Pros**: Scalable, powerful hardware
**Cons**: Network latency, cost

### Option 3: Hybrid
```
Camera → Edge (Fast Pass) ─┬→ Display
                           │
         Uncertain Cases ──┴→ Cloud (Deep Analysis)
```
**Pros**: Best of both worlds
**Cons**: More complex

## Technology Stack Justification

| Technology | Why Chosen | Alternative Considered |
|------------|------------|------------------------|
| **TensorFlow** | Industry standard, excellent docs | PyTorch (more research-oriented) |
| **Keras** | High-level API, fast prototyping | TF low-level API (more control) |
| **MobileNetV2** | Speed-accuracy balance | ResNet50 (slower), EfficientNet (newer) |
| **OpenCV** | Fast, mature, hardware acceleration | PIL (slower), scikit-image (simpler) |
| **Grad-CAM** | Visual, interpretable | LIME (slower), SHAP (complex) |

## Future Architecture Enhancements

1. **Model Versioning**
   - Track model versions
   - A/B testing
   - Rollback capability

2. **Distributed Training**
   - Multi-GPU training
   - Parameter server architecture
   - Faster iteration cycles

3. **AutoML Integration**
   - Neural architecture search
   - Hyperparameter tuning
   - Automated feature engineering

---

**Last Updated**: 2024
**Version**: 1.0.0