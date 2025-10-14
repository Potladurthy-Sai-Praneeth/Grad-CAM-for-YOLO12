import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from grad_cam import YOLO12GradCAM
from config import *
import argparse

def apply_heatmap(image, saliency_map, alpha=0.5):
    """
    Apply heatmap overlay on the original image
    
    Args:
        image: Original image (numpy array)
        saliency_map: Saliency map from GradCAM
        alpha: Transparency factor for overlay
    
    Returns:
        Overlaid image
    """
    # Convert image to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        # Handle batch dimension (1, C, H, W) -> (C, H, W)
        if image.ndim == 4:
            image = image[0]
        # Convert (C, H, W) to (H, W, C)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
    
    # Convert saliency map to numpy if needed
    if torch.is_tensor(saliency_map):
        saliency_map = saliency_map.cpu().numpy()
    
    # Handle batch dimension (1, 1, H, W) -> (1, H, W)
    if saliency_map.ndim == 4:
        saliency_map = saliency_map[0]
    
    # Handle channel dimension (1, H, W) -> (H, W)
    if saliency_map.ndim == 3 and saliency_map.shape[0] == 1:
        saliency_map = saliency_map[0]
    
    # Normalize image to 0-255 range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Handle grayscale images - convert to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    # Normalize saliency map to 0-1 range first
    saliency_min = saliency_map.min()
    saliency_max = saliency_map.max()
    if saliency_max > saliency_min:
        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
    else:
        saliency_map = np.zeros_like(saliency_map)
    
    # Convert to 0-255 range
    saliency_map = np.clip(saliency_map, 0, 1)
    saliency_map = np.uint8(255 * saliency_map)
    
    # Resize saliency map to match image size if needed
    if saliency_map.shape[:2] != image.shape[:2]:
        saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlaid, heatmap


def process_images(model_path, input_dir, output_dir):
    """
    Process all images in input_dir and save heatmaps to output_dir
    
    Args:
        model_path: Path to YOLO model
        input_dir: Directory containing input images organized by class
        output_dir: Directory to save output heatmaps
    
    Returns:
        Dictionary with sample images for visualization
    """
    # Initialize GradCAM
    gradcam = YOLO12GradCAM(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store one sample per class for final visualization
    class_samples = {}
    
    # Process each class folder
    for class_folder in sorted(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_folder)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"\nProcessing class: {class_folder}")
        
        # Create corresponding output folder
        output_class_path = os.path.join(output_dir, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Get all images in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process each image
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(class_path, img_name)
            gt_idx = label_class_mapping[img_path.split('/')[-1].split('_')[0]]
            
            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Prepare input tensor
            img_resized = cv2.resize(img_rgb, (256, 256))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # Generate GradCAM
            try:
                saliency_map = gradcam(img_tensor, gt_idx)
                print(f'Saliency map size is {saliency_map.shape}')

                if saliency_map is not None:
                    # Apply heatmap
                    overlaid, heatmap = apply_heatmap(img_resized, saliency_map)
                    
                    # Save overlaid image
                    output_path = os.path.join(output_class_path, img_name)
                    cv2.imwrite(output_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                    
                    # Save first image of each class for visualization
                    if class_folder not in class_samples:
                        class_samples[class_folder] = {
                            'original': img_resized,
                            'overlaid': overlaid,
                            'heatmap': heatmap
                        }
                    
                    print(f"  Processed {idx + 1}/{len(image_files)}: {img_name}")
                else:
                    print(f"  No detections for: {img_name}")
                    
            except Exception as e:
                print(f"  Error processing {img_name}: {str(e)}")
                continue
    
    return class_samples


def create_combined_visualization(class_samples, output_path):
    """
    Create a single visualization with one sample from each class
    Shows: original image | overlaid heatmap | heatmap only
    
    Args:
        class_samples: Dictionary of sample images per class
        output_path: Path to save the final visualization
    """
    num_classes = len(class_samples)
    
    if num_classes == 0:
        print("No samples to visualize!")
        return
    
    # Create figure with 3 columns (original, overlaid, heatmap) and rows for each class
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    
    # Handle case of single class
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    # Sort class names
    sorted_classes = sorted(class_samples.keys())
    
    for idx, class_name in enumerate(sorted_classes):
        sample = class_samples[class_name]
        
        # Original image
        axes[idx, 0].imshow(sample['original'])
        axes[idx, 0].set_title(f'{class_name}\n(Original)', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Overlaid image
        axes[idx, 1].imshow(sample['overlaid'])
        axes[idx, 1].set_title(f'{class_name}\n(GradCAM Overlay)', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Heatmap only
        axes[idx, 2].imshow(sample['heatmap'])
        axes[idx, 2].set_title(f'{class_name}\n(Heatmap)', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined visualization saved to: {output_path}")
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="GradCAM Visualization for YOLOv8")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--input', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output', type=str, required=True, help='Directory to save output images')
    args = parser.parse_args()

    VISUALIZATION_PATH = f'{args.output}/combined_visualization.png'
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found!")
        return
    
    print("="*60)
    print("GradCAM Heatmap Generation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Input Directory: {args.input}")
    print(f"Output Directory: {args.output}")
    print("="*60)
    
    # Process all images
    class_samples = process_images(args.model, args.input, args.output)
    
    # Create combined visualization
    if class_samples:
        create_combined_visualization(class_samples, VISUALIZATION_PATH)
        print("\n" + "="*60)
        print("Processing complete!")
        print(f"Output images saved to: {args.output}")
        print(f"Combined visualization: {VISUALIZATION_PATH}")
        print("="*60)
    else:
        print("\nNo images were processed successfully.")


if __name__ == "__main__":

    main()
