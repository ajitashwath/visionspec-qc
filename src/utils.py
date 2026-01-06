import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

def organize_images(source_dir, output_dir, train_split=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    for class_name in ['pass', 'defect']:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        train_imgs, val_imgs = train_test_split(
            images, 
            train_size=train_split, 
            random_state=42
        )
        
        train_dir = output_path / 'train' / class_name
        val_dir = output_path / 'validation' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir / img.name)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary)
    }
    return metrics

def visualize_predictions(images, predictions, labels, class_names=['Pass', 'Defect']):
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 4*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i])
        
        pred_class = class_names[1] if predictions[i] > 0.5 else class_names[0]
        true_class = class_names[1] if labels[i] == 1 else class_names[0]
        color = 'green' if pred_class == true_class else 'red'
        
        plt.title(f"Pred: {pred_class}\nTrue: {true_class}\n({predictions[i]:.2%})", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()