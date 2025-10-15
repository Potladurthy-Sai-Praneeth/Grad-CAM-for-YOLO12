"""
Inference Script for Leaf Disease Classification using YOLO-CLS
This script performs inference on test images and generates comprehensive classification results.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score
)
from collections import Counter, defaultdict


def load_model_and_config(model_path, dataset_yaml):
    """
    Load the trained YOLO classification model and dataset configuration
    
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
    Get list of test images from the dataset organized by class folders
    
    Args:
        dataset_config: Dataset configuration dictionary
    
    Returns:
        test_images: List of dictionaries with image paths and true labels
        class_names: List of class names
    """
    dataset_path = Path(dataset_config['path'])
    test_path = dataset_path / dataset_config.get('test', 'test')
    
    print(f"\nTest dataset path: {test_path}")
    print(f"Test path exists: {test_path.exists()}")
    
    test_images = []
    class_names = dataset_config['names']
    
    if test_path.exists():
        # Iterate through class directories
        for class_name in class_names:
            class_dir = test_path / class_name
            if class_dir.exists() and class_dir.is_dir():
                # Get all images in the class directory
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                class_images = []
                for ext in image_extensions:
                    class_images.extend(list(class_dir.glob(ext)))
                
                print(f"Class '{class_name}': {len(class_images)} images")
                
                # Store image info with true label
                for img_path in class_images:
                    test_images.append({
                        'path': img_path,
                        'true_label': class_name,
                        'true_class_id': class_names.index(class_name)
                    })
        
        print(f"\nTotal test images found: {len(test_images)}")
    else:
        print("Warning: Test path does not exist!")
    
    return test_images, class_names


def run_inference(model, test_images, class_names, top_k=5):
    """
    Run inference on all test images
    
    Args:
        model: YOLO classification model
        test_images: List of test image dictionaries
        class_names: List of class names
        top_k: Number of top predictions to store
    
    Returns:
        all_predictions: List of prediction dictionaries
        inference_times: List of inference times
    """
    print("\nRunning inference on test dataset...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_predictions = []
    inference_times = []
    
    for img_info in tqdm(test_images, desc="Processing images"):
        img_path = img_info['path']
        
        # Run prediction
        start_time = datetime.now()
        results = model.predict(
            source=str(img_path),
            verbose=False,
            device=device
        )
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        inference_times.append(inference_time)
        
        # Extract results for the first (and only) image
        result = results[0]
        
        # Get prediction probabilities
        probs = result.probs
        top_k_indices = probs.top5  # Get top 5 predictions
        top_k_confidences = probs.top5conf.cpu().numpy()
        
        # Store prediction data
        pred_data = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'true_label': img_info['true_label'],
            'true_class_id': img_info['true_class_id'],
            'inference_time': inference_time,
            'top_class_id': int(probs.top1),
            'top_class_name': class_names[int(probs.top1)],
            'top_confidence': float(probs.top1conf),
            'top_k_predictions': []
        }
        
        # Store top-k predictions
        for idx, (class_id, confidence) in enumerate(zip(top_k_indices, top_k_confidences)):
            pred_data['top_k_predictions'].append({
                'rank': idx + 1,
                'class_id': int(class_id),
                'class_name': class_names[int(class_id)],
                'confidence': float(confidence)
            })
        
        # Store all class probabilities
        all_probs = probs.data.cpu().numpy()
        pred_data['all_probabilities'] = {
            class_names[i]: float(all_probs[i]) for i in range(len(class_names))
        }
        
        all_predictions.append(pred_data)
    
    # Calculate inference statistics
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\nInference Statistics:")
    print(f"- Total images processed: {len(test_images)}")
    print(f"- Average inference time: {avg_inference_time:.4f} seconds")
    print(f"- FPS: {fps:.2f}")
    
    return all_predictions, inference_times


def calculate_classification_metrics(predictions, class_names):
    """
    Calculate comprehensive classification metrics
    
    Args:
        predictions: List of prediction dictionaries
        class_names: List of class names
    
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    print("\nCalculating classification metrics...")
    
    # Extract true labels and predictions
    y_true = [pred['true_class_id'] for pred in predictions]
    y_pred = [pred['top_class_id'] for pred in predictions]
    y_true_names = [pred['true_label'] for pred in predictions]
    y_pred_names = [pred['top_class_name'] for pred in predictions]
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics (weighted, macro, micro)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Additional metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    matthews_cc = matthews_corrcoef(y_true, y_pred)
    
    # Top-k accuracy
    top3_correct = sum(1 for pred in predictions 
                       if pred['true_class_id'] in [p['class_id'] for p in pred['top_k_predictions'][:3]])
    top3_accuracy = top3_correct / len(predictions)
    
    top5_correct = sum(1 for pred in predictions 
                       if pred['true_class_id'] in [p['class_id'] for p in pred['top_k_predictions'][:5]])
    top5_accuracy = top5_correct / len(predictions)
    
    # Confidence statistics
    correct_predictions = [pred for pred in predictions if pred['top_class_id'] == pred['true_class_id']]
    incorrect_predictions = [pred for pred in predictions if pred['top_class_id'] != pred['true_class_id']]
    
    avg_confidence_correct = np.mean([pred['top_confidence'] for pred in correct_predictions]) if correct_predictions else 0
    avg_confidence_incorrect = np.mean([pred['top_confidence'] for pred in incorrect_predictions]) if incorrect_predictions else 0
    avg_confidence_overall = np.mean([pred['top_confidence'] for pred in predictions])
    
    metrics_dict = {
        'Overall Metrics': {
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'Top-3 Accuracy': top3_accuracy,
            'Top-5 Accuracy': top5_accuracy,
            'Cohen Kappa Score': cohen_kappa,
            'Matthews Correlation Coefficient': matthews_cc,
        },
        'Precision': {
            'Macro': precision_macro,
            'Micro': precision_micro,
            'Weighted': precision_weighted,
        },
        'Recall': {
            'Macro': recall_macro,
            'Micro': recall_micro,
            'Weighted': recall_weighted,
        },
        'F1-Score': {
            'Macro': f1_macro,
            'Micro': f1_micro,
            'Weighted': f1_weighted,
        },
        'Confidence Statistics': {
            'Average (Overall)': avg_confidence_overall,
            'Average (Correct)': avg_confidence_correct,
            'Average (Incorrect)': avg_confidence_incorrect,
            'Confidence Gap': avg_confidence_correct - avg_confidence_incorrect,
        }
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_metrics = []
    for i, class_name in enumerate(class_names):
        per_class_metrics.append({
            'Class': class_name,
            'Precision': per_class_precision[i],
            'Recall': per_class_recall[i],
            'F1-Score': per_class_f1[i],
            'Support': sum(1 for y in y_true if y == i)
        })
    
    metrics_dict['Per-Class Metrics'] = per_class_metrics
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics_dict['Confusion Matrix'] = cm
    
    return metrics_dict


def print_metrics(metrics_dict):
    """
    Print all metrics in a formatted way
    
    Args:
        metrics_dict: Dictionary containing all metrics
    """
    print("\n" + "="*70)
    print(" "*20 + "CLASSIFICATION METRICS")
    print("="*70)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    for metric, value in metrics_dict['Overall Metrics'].items():
        print(f"   {metric:40s}: {value:.4f}")
    
    # Aggregated metrics
    for metric_type in ['Precision', 'Recall', 'F1-Score']:
        print(f"\n📈 {metric_type}:")
        for avg_type, value in metrics_dict[metric_type].items():
            print(f"   {avg_type:40s}: {value:.4f}")
    
    # Confidence statistics
    print("\n🎯 Confidence Statistics:")
    for stat, value in metrics_dict['Confidence Statistics'].items():
        print(f"   {stat:40s}: {value:.4f}")
    
    # Per-class metrics
    print("\n📋 Per-Class Metrics:")
    per_class_df = pd.DataFrame(metrics_dict['Per-Class Metrics'])
    print(per_class_df.to_string(index=False))


def save_metrics(metrics_dict, output_dir):
    """
    Save metrics to files
    
    Args:
        metrics_dict: Dictionary containing all metrics
        output_dir: Directory to save metrics
    """
    print("\nSaving metrics to files...")
    
    # Save overall metrics
    overall_data = []
    for category in ['Overall Metrics', 'Precision', 'Recall', 'F1-Score', 'Confidence Statistics']:
        for metric, value in metrics_dict[category].items():
            overall_data.append({
                'Category': category,
                'Metric': metric,
                'Value': value
            })
    
    overall_df = pd.DataFrame(overall_data)
    overall_path = os.path.join(output_dir, 'overall_metrics.csv')
    overall_df.to_csv(overall_path, index=False)
    print(f"Overall metrics saved to: {overall_path}")
    
    # Save per-class metrics
    per_class_df = pd.DataFrame(metrics_dict['Per-Class Metrics'])
    per_class_path = os.path.join(output_dir, 'per_class_metrics.csv')
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Per-class metrics saved to: {per_class_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        metrics_dict['Confusion Matrix'],
        index=[m['Class'] for m in metrics_dict['Per-Class Metrics']],
        columns=[m['Class'] for m in metrics_dict['Per-Class Metrics']]
    )
    cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


def visualize_confusion_matrix(metrics_dict, class_names, output_dir):
    """
    Visualize confusion matrix as a heatmap
    
    Args:
        metrics_dict: Dictionary containing all metrics
        class_names: List of class names
        output_dir: Directory to save visualization
    """
    cm = metrics_dict['Confusion Matrix']
    
    # Calculate normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix visualization saved to: {save_path}")
    plt.close()


def visualize_per_class_metrics(metrics_dict, output_dir):
    """
    Visualize per-class metrics
    
    Args:
        metrics_dict: Dictionary containing all metrics
        output_dir: Directory to save visualization
    """
    per_class_df = pd.DataFrame(metrics_dict['Per-Class Metrics'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'Support']
    colors = sns.color_palette("husl", len(per_class_df))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        if metric == 'Support':
            # Bar chart for support
            ax.bar(range(len(per_class_df)), per_class_df[metric], color=colors)
            ax.set_ylabel('Number of Samples', fontsize=11)
        else:
            # Horizontal bar chart for performance metrics
            ax.barh(range(len(per_class_df)), per_class_df[metric], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Score', fontsize=11)
            
            # Add value labels
            for i, v in enumerate(per_class_df[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        ax.set_title(f'{metric} by Class', fontsize=12, fontweight='bold')
        
        if metric == 'Support':
            ax.set_xticks(range(len(per_class_df)))
            ax.set_xticklabels(per_class_df['Class'], rotation=45, ha='right')
        else:
            ax.set_yticks(range(len(per_class_df)))
            ax.set_yticklabels(per_class_df['Class'])
        
        ax.grid(axis='x' if metric != 'Support' else 'y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Per-class metrics visualization saved to: {save_path}")
    plt.close()


def visualize_confidence_distribution(predictions, output_dir):
    """
    Visualize confidence score distributions
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save visualization
    """
    correct_predictions = [pred for pred in predictions if pred['top_class_id'] == pred['true_class_id']]
    incorrect_predictions = [pred for pred in predictions if pred['top_class_id'] != pred['true_class_id']]
    
    correct_confidences = [pred['top_confidence'] for pred in correct_predictions]
    incorrect_confidences = [pred['top_confidence'] for pred in incorrect_predictions]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [correct_confidences, incorrect_confidences]
    bp = ax2.boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                     patch_artist=True, notch=True)
    
    # Color the boxes
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Confidence Score Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confidence distribution visualization saved to: {save_path}")
    plt.close()


def visualize_sample_predictions(predictions, output_dir, num_samples=12):
    """
    Visualize sample predictions with images
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save visualization
        num_samples: Number of samples to visualize
    """
    # Get correct and incorrect predictions
    correct_preds = [p for p in predictions if p['top_class_id'] == p['true_class_id']]
    incorrect_preds = [p for p in predictions if p['top_class_id'] != p['true_class_id']]
    
    # Select samples (half correct, half incorrect)
    num_correct = min(num_samples // 2, len(correct_preds))
    num_incorrect = min(num_samples // 2, len(incorrect_preds))
    
    if num_correct < num_samples // 2:
        num_incorrect = min(num_samples - num_correct, len(incorrect_preds))
    elif num_incorrect < num_samples // 2:
        num_correct = min(num_samples - num_incorrect, len(correct_preds))
    
    selected_correct = np.random.choice(len(correct_preds), num_correct, replace=False) if correct_preds else []
    selected_incorrect = np.random.choice(len(incorrect_preds), num_incorrect, replace=False) if incorrect_preds else []
    
    samples = []
    samples.extend([correct_preds[i] for i in selected_correct])
    samples.extend([incorrect_preds[i] for i in selected_incorrect])
    
    # Create visualization
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]
    
    for idx, pred in enumerate(samples):
        img_path = pred['image_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Determine if correct or incorrect
        is_correct = pred['top_class_id'] == pred['true_class_id']
        border_color = 'green' if is_correct else 'red'
        
        # Create title
        title = f"True: {pred['true_label']}\n"
        title += f"Pred: {pred['top_class_name']} ({pred['top_confidence']:.3f})"
        
        axes[idx].set_title(title, fontsize=10, 
                           color=border_color, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    # Hide empty subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions visualization saved to: {save_path}")
    plt.close()


def analyze_misclassifications(predictions, class_names, output_dir):
    """
    Analyze and visualize misclassification patterns
    
    Args:
        predictions: List of prediction dictionaries
        class_names: List of class names
        output_dir: Directory to save analysis
    """
    # Get misclassified samples
    misclassified = [p for p in predictions if p['top_class_id'] != p['true_class_id']]
    
    if not misclassified:
        print("No misclassifications found!")
        return
    
    print(f"\nAnalyzing {len(misclassified)} misclassifications...")
    
    # Create misclassification matrix (true -> predicted)
    misclass_matrix = defaultdict(lambda: defaultdict(int))
    
    for pred in misclassified:
        true_label = pred['true_label']
        pred_label = pred['top_class_name']
        misclass_matrix[true_label][pred_label] += 1
    
    # Create detailed misclassification report
    misclass_data = []
    for pred in misclassified:
        misclass_data.append({
            'Image': pred['image_name'],
            'True Label': pred['true_label'],
            'Predicted Label': pred['top_class_name'],
            'Confidence': pred['top_confidence'],
            'Top-2': pred['top_k_predictions'][1]['class_name'] if len(pred['top_k_predictions']) > 1 else 'N/A',
            'Top-3': pred['top_k_predictions'][2]['class_name'] if len(pred['top_k_predictions']) > 2 else 'N/A',
        })
    
    misclass_df = pd.DataFrame(misclass_data)
    misclass_path = os.path.join(output_dir, 'misclassifications.csv')
    misclass_df.to_csv(misclass_path, index=False)
    print(f"Misclassification details saved to: {misclass_path}")
    
    # Visualize most common misclassification pairs
    misclass_pairs = []
    for true_label, pred_dict in misclass_matrix.items():
        for pred_label, count in pred_dict.items():
            misclass_pairs.append({
                'pair': f'{true_label} → {pred_label}',
                'count': count
            })
    
    misclass_pairs_df = pd.DataFrame(misclass_pairs).sort_values('count', ascending=False).head(10)
    
    if not misclass_pairs_df.empty:
        plt.figure(figsize=(12, 6))
        plt.barh(misclass_pairs_df['pair'], misclass_pairs_df['count'],
                color=sns.color_palette("Reds_r", len(misclass_pairs_df)))
        plt.xlabel('Number of Misclassifications', fontsize=12)
        plt.title('Top 10 Misclassification Patterns', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'misclassification_patterns.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassification patterns visualization saved to: {save_path}")
        plt.close()


def save_predictions(predictions, output_dir):
    """
    Save predictions to various formats
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save predictions
    """
    print("\nSaving predictions...")
    
    # Save to JSON
    json_path = os.path.join(output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to JSON: {json_path}")
    
    # Create CSV with main predictions
    csv_data = []
    for pred in predictions:
        csv_data.append({
            'Image Name': pred['image_name'],
            'Image Path': pred['image_path'],
            'True Label': pred['true_label'],
            'Predicted Label': pred['top_class_name'],
            'Confidence': pred['top_confidence'],
            'Correct': pred['top_class_id'] == pred['true_class_id'],
            'Inference Time (s)': pred['inference_time']
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'predictions.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to CSV: {csv_path}")


def generate_summary_report(predictions, metrics_dict, inference_times, output_dir):
    """
    Generate a comprehensive summary report
    
    Args:
        predictions: List of prediction dictionaries
        metrics_dict: Dictionary containing all metrics
        inference_times: List of inference times
        output_dir: Directory to save report
    """
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    correct = sum(1 for p in predictions if p['top_class_id'] == p['true_class_id'])
    total = len(predictions)
    
    report = f"""
{'='*70}
                CLASSIFICATION INFERENCE REPORT
{'='*70}

Dataset Statistics:
- Total Images: {total}
- Correctly Classified: {correct} ({100*correct/total:.2f}%)
- Incorrectly Classified: {total - correct} ({100*(total-correct)/total:.2f}%)

Performance Metrics:
- Accuracy: {metrics_dict['Overall Metrics']['Accuracy']:.4f}
- Balanced Accuracy: {metrics_dict['Overall Metrics']['Balanced Accuracy']:.4f}
- Top-3 Accuracy: {metrics_dict['Overall Metrics']['Top-3 Accuracy']:.4f}
- Top-5 Accuracy: {metrics_dict['Overall Metrics']['Top-5 Accuracy']:.4f}
- Macro F1-Score: {metrics_dict['F1-Score']['Macro']:.4f}
- Weighted F1-Score: {metrics_dict['F1-Score']['Weighted']:.4f}
- Cohen Kappa: {metrics_dict['Overall Metrics']['Cohen Kappa Score']:.4f}
- Matthews CC: {metrics_dict['Overall Metrics']['Matthews Correlation Coefficient']:.4f}

Inference Performance:
- Average Inference Time: {avg_inference_time:.4f} seconds
- Throughput: {fps:.2f} FPS
- Total Processing Time: {sum(inference_times):.2f} seconds

Confidence Statistics:
- Overall Average: {metrics_dict['Confidence Statistics']['Average (Overall)']:.4f}
- Correct Predictions: {metrics_dict['Confidence Statistics']['Average (Correct)']:.4f}
- Incorrect Predictions: {metrics_dict['Confidence Statistics']['Average (Incorrect)']:.4f}
- Confidence Gap: {metrics_dict['Confidence Statistics']['Confidence Gap']:.4f}

{'='*70}
"""
    
    print(report)
    
    # Save report to file
    report_path = os.path.join(output_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Summary report saved to: {report_path}")


def main():
    """Main inference function for classification"""
    parser = argparse.ArgumentParser(description='Inference script for Leaf Disease Classification')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--output', type=str, default='cls_test_results', help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=12, help='Number of sample visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print(" "*15 + "YOLO CLASSIFICATION INFERENCE")
    print("="*70)
    
    # Load model and configuration
    model, dataset_config, class_names, num_classes = load_model_and_config(
        args.model, args.data
    )
    
    # Get test images
    test_images, class_names = get_test_images(dataset_config)
    
    if len(test_images) == 0:
        print("Error: No test images found!")
        return
    
    # Run inference
    all_predictions, inference_times = run_inference(
        model, test_images, class_names
    )
    
    # Calculate metrics
    metrics_dict = calculate_classification_metrics(all_predictions, class_names)
    
    # Print metrics
    print_metrics(metrics_dict)
    
    # Save metrics
    save_metrics(metrics_dict, args.output)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_confusion_matrix(metrics_dict, class_names, args.output)
    visualize_per_class_metrics(metrics_dict, args.output)
    visualize_confidence_distribution(all_predictions, args.output)
    visualize_sample_predictions(all_predictions, args.output, num_samples=args.num_samples)
    
    # Analyze misclassifications
    analyze_misclassifications(all_predictions, class_names, args.output)
    
    # Save predictions
    save_predictions(all_predictions, args.output)
    
    # Generate summary report
    generate_summary_report(all_predictions, metrics_dict, inference_times, args.output)
    
    print("\n" + "="*70)
    print(" "*20 + "INFERENCE COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {args.output}")
    print("\n✅ Generated files:")
    print("   - overall_metrics.csv")
    print("   - per_class_metrics.csv")
    print("   - confusion_matrix.csv & .png")
    print("   - per_class_metrics.png")
    print("   - confidence_distribution.png")
    print("   - sample_predictions.png")
    print("   - misclassifications.csv")
    print("   - misclassification_patterns.png")
    print("   - predictions.json & .csv")
    print("   - summary_report.txt")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
