# Grad-CAM for Explainable YOLO-based Tomato Leaf Disease Identification

## 🔍 Overview

This project implements an **Explainable AI (XAI)** system for tomato leaf disease detection using state-of-the-art YOLO (You Only Look Once) models with **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualization. The primary focus is on making deep learning models interpretable by highlighting which regions of leaf images contribute most to disease classification decisions.

### Disease Classes

The system can detect **10 different tomato leaf conditions**:

## ✨ Key Features

- **🎯 Dual Model Architecture**: Supports both bounding box detection and classification modes
- **🔬 Explainable AI**: Integrated Grad-CAM visualization for model interpretability
- **⚡ High Performance**: Built on YOLOv12 architecture for fast and accurate predictions
- **📊 Comprehensive Visualization**: Generates heatmaps showing disease-affected regions

## 🧠 Grad-CAM Explainability

### What is Grad-CAM?

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainability technique that produces visual explanations for decisions from CNN-based models. It highlights the important regions in the image that influenced the model's prediction.

### How Grad-CAM Works

This implementation (`grad_cam.py`) provides a custom Grad-CAM solution tailored for YOLO models [2,5]:

1. **Activation and Gradient Extraction**
2. **Forward Pass with Gradient Tracking**
3. **Gradient Backpropagation**
4. **Neuron Importance Weighting**
5. **Weighted Activation Maps**
6. **ReLU and Upsampling**
7. **Multi-Layer Aggregation**

### Advantages of Grad-CAM for YOLO

1. **Class-Discriminative**: Shows regions specific to each disease class
2. **High Resolution**: Provides fine-grained localization of affected areas
3. **No Retraining Required**: Works with already-trained models
4. **Computationally Efficient**: Fast enough for real-time applications
5. **Interpretable**: Easy for plant pathologists to validate model decisions

### Output Interpretation

The Grad-CAM heatmap uses color coding:
- 🔴 **Red/Yellow regions**: High importance - areas the model focuses on for disease detection
- 🟢 **Green/Blue regions**: Lower importance - less relevant for the prediction
- 🔵 **Dark blue**: Background or irrelevant areas


## 🛠️ Installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Potladurthy-Sai-Praneeth/Leaf-disease-prediction.git
cd Leaf-disease-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training

#### Classification Model (Recommended for Grad-CAM)
```bash
python train.py --mode cls --model_size n --dataset ./yolo_cls_dataset --output ./yolo_training
```

#### Bounding Box Detection Model
```bash
python train.py --mode bbox --model_size n --dataset ./dataset/classes.yaml --output ./yolo_training
```

**Arguments:**
- `--mode`: Training mode (`cls` for classification, `bbox` for detection)
- `--model_size`: YOLO model size (`n`, `s`, `m`, `l`, `x`)
- `--dataset`: Path to dataset (directory for cls, YAML for bbox)
- `--output`: Output directory for trained models

### Inference

#### Classification
```bash
python inference_cls.py --model ./best_cls.pt --data ./dataset/classes.yaml
```

#### Bounding Box Detection
```bash
python inference_bbox.py --model ./best.pt --data ./dataset/classes.yaml
```

### Grad-CAM Visualization

The Grad-CAM visualization script generates explainability heatmaps that show which regions of the leaf images the model focuses on when making predictions.

#### Basic Usage
```bash
python run_grad_cam.py \
    --model ./best_cls.pt \
    --input ./grad_cam/input_images \
    --output ./grad_cam/output_images \
    --mode cls
```

#### Parameters

- `--model` (required): Path to the trained YOLO model (`.pt` file)
  - Use `best_cls.pt` for classification models
  - Use `best.pt` for detection models

- `--input` (required): Directory containing input images
  - Images should be organized in class-specific subdirectories
  - Example: `grad_cam/input_images/Healthy/`, `grad_cam/input_images/Bacterial_spot/`

- `--output` (required): Directory where Grad-CAM visualizations will be saved
  - Output structure mirrors input directory structure
  - Includes individual heatmaps and combined visualization

- `--mode` (optional, default: `cls`): Model training mode
  - `cls`: Classification model (recommended for interpretability)
  - `box`: Bounding box detection model

## 📈 Results

### Sample Grad-CAM Visualization

![Combined Grad-CAM Visualization](grad_cam/output_images/combined_visualization.png)


## 📚 References

1. **Grad-CAM Paper**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. **YOLO**: Ultralytics YOLOv12 - https://github.com/ultralytics/ultralytics
3. **Plant Disease Detection**: Hughes & Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics", arXiv 2015
4. **Kaggle Dataset**: https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo
5. **PyTorch Grad-CAM**: https://github.com/jacobgil/pytorch-grad-cam