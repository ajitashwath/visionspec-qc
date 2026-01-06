import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/mobile_net_v2_pcb.h5'
CONFIDENCE_THRESHOLD = 0.5
LAST_CONV_LAYER = 'out_relu'

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded")

def generate_gradcam(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam_overlay(frame, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Starting inference... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = cv2.resize(frame, (224, 224))
    input_array = np.expand_dims(input_frame, axis=0) / 255.0
    prediction = model.predict(input_array, verbose=0)[0][0]

    display_frame = frame.copy()
    
    if prediction > CONFIDENCE_THRESHOLD:
        label = "DEFECT"
        color = (0, 0, 255)
        heatmap = generate_gradcam(input_array, model, LAST_CONV_LAYER)
        display_frame = apply_gradcam_overlay(display_frame, heatmap)
    else:
        label = "PASS"
        color = (0, 255, 0)  
    
    text = f"{label}: {prediction:.2%}"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('VisionSpec QC', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Inference stopped")