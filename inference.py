"""
Inference Script for Leaf Disease Detection using YOLOv12
This script performs inference on test images and generates comprehensive results.
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import argparse


def load_model_and_config(model_path, dataset_yaml):
    """
    Load the trained YOLO model and dataset configuration
    
    Args:
        model_path: Path to the trained model weights
        dataset_yaml: Path to the dataset YAML configuration
    
    Returns:
        model: Loaded YOLO model
        dataset_config: Dataset configuration dictionary
        class_names: List of class names
        num_classes: Number of classes
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model loaded successfully on device: {device}")
    print(f"Model Info:")
    print(f"- Task: {model.task}")
    print(f"- Model type: {type(model.model)}")
    
    # Load dataset configuration
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("\nDataset Configuration:")
    print(f"- Dataset path: {dataset_config.get('path', 'N/A')}")
    print(f"- Number of classes: {dataset_config.get('nc', 'N/A')}")
    print(f"- Class names: {dataset_config.get('names', 'N/A')}")
    
    class_names = dataset_config['names']
    num_classes = dataset_config['nc']
    
    return model, dataset_config, class_names, num_classes


def get_test_images(dataset_config):
    """
    Get list of test images from the dataset
    
    Args:
        dataset_config: Dataset configuration dictionary
    
    Returns:
        test_images: List of paths to test images
    """
    dataset_path = Path(dataset_config['path'])
    test_path = dataset_path / dataset_config.get('test', 'test/images')
    
    print(f"\nTest dataset path: {test_path}")
    print(f"Test path exists: {test_path.exists()}")
    
    if test_path.exists():
        test_images = list(test_path.glob('*.jpg')) 
        print(f"Number of test images found: {len(test_images)}")
    else:
        print("Warning: Test path does not exist!")
        test_images = []
    
    return test_images


def run_inference(model, test_images, class_names, conf_threshold=0.5, 
                 iou_threshold=0.5, imgsz=256):
    """
    Run inference on all test images
    
    Args:
        model: YOLO model
        test_images: List of test image paths
        class_names: List of class names
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
    
    Returns:
        all_predictions: List of prediction dictionaries
        inference_times: List of inference times
    """
    print("\nRunning inference on test dataset...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_predictions = []
    inference_times = []
    
    for img_path in tqdm(test_images, desc="Processing images"):
        # Run prediction
        start_time = datetime.now()
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device=device
        )
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        inference_times.append(inference_time)
        
        # Extract results for the first (and only) image
        result = results[0]
        
        # Store prediction data
        pred_data = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'inference_time': inference_time,
            'detections': []
        }
        
        # Extract detection information
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Extract masks if available
            masks = None
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            
            # Store each detection
            for i in range(len(boxes)):
                detection = {
                    'box': boxes[i].tolist(),
                    'confidence': float(confidences[i]),
                    'class_id': int(classes[i]),
                    'class_name': class_names[int(classes[i])],
                    'has_mask': masks is not None
                }
                pred_data['detections'].append(detection)
        
        all_predictions.append(pred_data)
    
    # Calculate inference statistics
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\nInference Statistics:")
    print(f"- Total images processed: {len(test_images)}")
    print(f"- Average inference time: {avg_inference_time:.4f} seconds")
    print(f"- FPS: {fps:.2f}")
    print(f"- Total detections: {sum(len(p['detections']) for p in all_predictions)}")
    
    return all_predictions, inference_times


def visualize_predictions(model, predictions, output_dir, num_samples=6,
                         conf_threshold=0.5, iou_threshold=0.5, imgsz=256):
    """
    Visualize predictions with bounding boxes and segmentation masks
    
    Args:
        model: YOLO model
        predictions: List of prediction dictionaries
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Filter predictions with at least one detection
    preds_with_detections = [p for p in predictions if len(p['detections']) > 0]
    
    if len(preds_with_detections) == 0:
        print("No detections found in the predictions!")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(preds_with_detections))
    sample_indices = np.random.choice(len(preds_with_detections), num_samples, replace=False)
    
    # Create subplot grid
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, sample_idx in enumerate(sample_indices):
        pred = preds_with_detections[sample_idx]
        img_path = pred['image_path']
        
        # Run prediction again to get visualized image
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device=device
        )
        
        # Get the plotted image
        plotted_img = results[0].plot()
        plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx].imshow(plotted_img)
        axes[idx].axis('off')
        
        # Add title with detection info
        num_detections = len(pred['detections'])
        detected_classes = [d['class_name'] for d in pred['detections']]
        title = f"{pred['image_name']}\n{num_detections} detections: {', '.join(set(detected_classes))}"
        axes[idx].set_title(title, fontsize=10)
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions visualization saved to: {save_path}")
    plt.close()


def visualize_single_prediction(model, pred_data, output_dir,
                               conf_threshold=0.5, iou_threshold=0.5, imgsz=256):
    """
    Create a detailed visualization for a single prediction
    
    Args:
        model: YOLO model
        pred_data: Prediction data dictionary
        output_dir: Directory to save visualization
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_path = pred_data['image_path']
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Prediction with annotations
    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
        device=device
    )
    plotted_img = results[0].plot()
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    
    ax2.imshow(plotted_img)
    ax2.set_title(f'Predictions ({len(pred_data["detections"])} detections)', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add detection details as text
    detection_text = f"Image: {pred_data['image_name']}\n"
    detection_text += f"Inference time: {pred_data['inference_time']:.4f}s\n\n"
    detection_text += "Detections:\n"
    
    for i, det in enumerate(pred_data['detections'], 1):
        detection_text += f"{i}. {det['class_name']} ({det['confidence']:.2f})\n"
    
    fig.text(0.5, 0.02, detection_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'detailed_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Detailed prediction visualization saved to: {save_path}")
    plt.close()


def calculate_metrics(model, dataset_yaml, output_dir, imgsz=256,
                     conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate performance metrics on the test dataset
    
    Args:
        model: YOLO model
        dataset_yaml: Path to dataset YAML
        output_dir: Directory to save metrics
        imgsz: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        val_results: Validation results object
    """
    print("\nRunning validation to calculate performance metrics...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run validation on the test dataset
    val_results = model.val(
        data=dataset_yaml,
        split='test',
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=True
    )
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    
    # Extract and display metrics
    metrics_dict = {
        'Metric': [],
        'Value': []
    }
    
    # Box metrics
    if hasattr(val_results, 'box'):
        box_metrics = val_results.box
        metrics_dict['Metric'].extend([
            'Box mAP@0.5',
            'Box mAP@0.5:0.95',
            'Box Precision',
            'Box Recall'
        ])
        metrics_dict['Value'].extend([
            f"{box_metrics.map50:.4f}",
            f"{box_metrics.map:.4f}",
            f"{box_metrics.mp:.4f}",
            f"{box_metrics.mr:.4f}"
        ])
    
    # Mask metrics (for segmentation)
    if hasattr(val_results, 'seg'):
        seg_metrics = val_results.seg
        metrics_dict['Metric'].extend([
            'Mask mAP@0.5',
            'Mask mAP@0.5:0.95',
            'Mask Precision',
            'Mask Recall'
        ])
        metrics_dict['Value'].extend([
            f"{seg_metrics.map50:.4f}",
            f"{seg_metrics.map:.4f}",
            f"{seg_metrics.mp:.4f}",
            f"{seg_metrics.mr:.4f}"
        ])
    
    # Create DataFrame and save
    metrics_df = pd.DataFrame(metrics_dict)
    print("\nOverall Metrics:")
    print(metrics_df.to_string(index=False))
    
    metrics_path = os.path.join(output_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return val_results


def visualize_class_performance(val_results, class_names, output_dir):
    """
    Visualize per-class performance metrics
    
    Args:
        val_results: Validation results object
        class_names: List of class names
        output_dir: Directory to save visualizations
    """
    if not hasattr(val_results, 'box'):
        print("No box metrics available for class performance visualization")
        return
    
    # Get per-class metrics
    class_metrics = []
    
    for i, class_name in enumerate(class_names):
        if hasattr(val_results.box, 'ap_class_index'):
            # Find metrics for this class
            class_idx = None
            if i in val_results.box.ap_class_index:
                class_idx = list(val_results.box.ap_class_index).index(i)
            
            if class_idx is not None:
                class_metrics.append({
                    'Class': class_name,
                    'Precision': val_results.box.p[class_idx] if hasattr(val_results.box, 'p') else 0,
                    'Recall': val_results.box.r[class_idx] if hasattr(val_results.box, 'r') else 0,
                    'mAP@0.5': val_results.box.ap50[class_idx] if hasattr(val_results.box, 'ap50') else 0,
                    'mAP@0.5:0.95': val_results.box.ap[class_idx] if hasattr(val_results.box, 'ap') else 0
                })
    
    if not class_metrics:
        print("No class metrics available")
        return
    
    class_df = pd.DataFrame(class_metrics)
    print("\nPer-Class Metrics:")
    print(class_df.to_string(index=False))
    
    # Save per-class metrics
    class_metrics_path = os.path.join(output_dir, 'class_metrics.csv')
    class_df.to_csv(class_metrics_path, index=False)
    
    # Visualize per-class performance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        ax.barh(class_df['Class'], class_df[metric], 
                color=sns.color_palette("husl", len(class_df)))
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} by Class', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(class_df[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_performance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class performance visualization saved to: {save_path}")
    plt.close()


def save_results(model, predictions, output_dir, conf_threshold=0.5,
                iou_threshold=0.5, imgsz=256):
    """
    Save all prediction results to various formats
    
    Args:
        model: YOLO model
        predictions: List of prediction dictionaries
        output_dir: Directory to save results
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create subdirectories
    annotated_images_dir = os.path.join(output_dir, 'annotated_images')
    os.makedirs(annotated_images_dir, exist_ok=True)
    
    print("\nSaving prediction results...")
    
    # Save annotated images
    print("Saving annotated images...")
    for pred in tqdm(predictions, desc="Saving annotated images"):
        if len(pred['detections']) > 0:
            img_path = pred['image_path']
            
            # Run prediction to get annotated image
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False,
                device=device
            )
            
            # Save annotated image
            annotated_img = results[0].plot()
            output_path = os.path.join(annotated_images_dir, pred['image_name'])
            cv2.imwrite(output_path, annotated_img)
    
    print(f"Annotated images saved to: {annotated_images_dir}")
    
    # Save predictions to JSON
    json_output_path = os.path.join(output_dir, 'predictions.json')
    with open(json_output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to JSON: {json_output_path}")
    
    # Create comprehensive CSV report
    csv_data = []
    for pred in predictions:
        if len(pred['detections']) > 0:
            for det in pred['detections']:
                csv_data.append({
                    'image_name': pred['image_name'],
                    'image_path': pred['image_path'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'bbox_x1': det['box'][0],
                    'bbox_y1': det['box'][1],
                    'bbox_x2': det['box'][2],
                    'bbox_y2': det['box'][3],
                    'has_mask': det['has_mask'],
                    'inference_time': pred['inference_time']
                })
        else:
            # Record images with no detections
            csv_data.append({
                'image_name': pred['image_name'],
                'image_path': pred['image_path'],
                'class_id': None,
                'class_name': 'No Detection',
                'confidence': 0,
                'bbox_x1': None,
                'bbox_y1': None,
                'bbox_x2': None,
                'bbox_y2': None,
                'has_mask': False,
                'inference_time': pred['inference_time']
            })
    
    predictions_df = pd.DataFrame(csv_data)
    csv_output_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to CSV: {csv_output_path}")
    print(f"Total rows in CSV: {len(predictions_df)}")


def generate_summary_statistics(predictions, inference_times, output_dir):
    """
    Generate and save summary statistics
    
    Args:
        predictions: List of prediction dictionaries
        inference_times: List of inference times
        output_dir: Directory to save statistics
    """
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print("\n" + "="*50)
    print("PREDICTION SUMMARY STATISTICS")
    print("="*50)
    
    summary_stats = {
        'Total Images': len(predictions),
        'Images with Detections': len([p for p in predictions if len(p['detections']) > 0]),
        'Images without Detections': len([p for p in predictions if len(p['detections']) == 0]),
        'Total Detections': sum(len(p['detections']) for p in predictions),
        'Average Detections per Image': np.mean([len(p['detections']) for p in predictions]),
        'Average Confidence': np.mean([d['confidence'] for p in predictions for d in p['detections']]) if sum(len(p['detections']) for p in predictions) > 0 else 0,
        'Average Inference Time (s)': avg_inference_time,
        'FPS': fps
    }
    
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Statistic', 'Value'])
    print(summary_df.to_string(index=False))
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")

def print_final_summary(predictions, test_images, inference_times, 
                       val_results, output_dir):
    """
    Print final summary of inference
    
    Args:
        predictions: List of prediction dictionaries
        test_images: List of test image paths
        inference_times: List of inference times
        val_results: Validation results object
        output_dir: Output directory
    """
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print("\n" + "="*70)
    print(" "*20 + "INFERENCE COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"\n📊 Summary:")
    print(f"   - Processed {len(test_images)} images")
    print(f"   - Found {sum(len(p['detections']) for p in predictions)} detections")
    print(f"   - Average inference time: {avg_inference_time:.4f}s ({fps:.2f} FPS)")
    
    if hasattr(val_results, 'box'):
        print(f"   - Box mAP@0.5: {val_results.box.map50:.4f}")
        print(f"   - Box mAP@0.5:0.95: {val_results.box.map:.4f}")
    
    if hasattr(val_results, 'seg'):
        print(f"   - Mask mAP@0.5: {val_results.seg.map50:.4f}")
        print(f"   - Mask mAP@0.5:0.95: {val_results.seg.map:.4f}")
    
    print("\n" + "="*70)


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Inference script for Leaf Disease Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=256, help='Image size for inference')
    parser.add_argument('--num-samples', type=int, default=6, help='Number of sample visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and configuration
    model, dataset_config, class_names, num_classes = load_model_and_config(
        args.model, args.data
    )
    
    # Get test images
    test_images = get_test_images(dataset_config)
    
    if len(test_images) == 0:
        print("Error: No test images found!")
        return
    
    # Run inference
    all_predictions, inference_times = run_inference(
        model, test_images, class_names, 
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    visualize_predictions(
        model, all_predictions, args.output,
        num_samples=args.num_samples,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )
    
    # Visualize detailed prediction for one image
    preds_with_detections = [p for p in all_predictions if len(p['detections']) > 0]
    if len(preds_with_detections) > 0:
        visualize_single_prediction(
            model, preds_with_detections[0], args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz
        )
    
    # Calculate metrics
    val_results = calculate_metrics(
        model, args.data, args.output,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Visualize class performance
    visualize_class_performance(val_results, class_names, args.output)
    
    # Save results
    save_results(
        model, all_predictions, args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )
    
    # Generate summary statistics
    generate_summary_statistics(all_predictions, inference_times, args.output)
    
    # Print final summary
    print_final_summary(
        all_predictions, test_images, inference_times,
        val_results, args.output
    )


if __name__ == '__main__':
    main()
