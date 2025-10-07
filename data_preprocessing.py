import pandas as pd
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from config import *

class YOLODataProcessor:
    def __init__(self, dataset_path,output_path):
        """
        Initialize YOLO data processor
        
        Args:
            dataset_path: Path to the parquet file containing annotations
            processed_dir: Directory to save processed YOLO dataset
        """
        assert dataset_path is not None, "Dataset path must be provided"
        assert output_path is not None, "Output path must be provided"
        
        self.dataset = Path(dataset_path)
        self.processed_dir = Path(output_path)

        self.label_class_mapping = label_class_mapping
        self.num_classes = NUM_CLASSES
        self.class_name_mapping = class_name_mapping
        self.label_name_mapping = label_name_mapping
        self.imgsz = IMGSZ

        self.setup_augmentations()
        self.clean_labels(self.dataset, self.label_class_mapping)
    
    def setup_directories(self):
        """
        Create YOLO dataset directory structure.
        As the dataset is limited we can use CPU to perform augmentations and store the images locally.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(self.processed_dir / 'images' / split, exist_ok=True)
            os.makedirs(self.processed_dir / 'labels' / split, exist_ok=True)


    def setup_augmentations(self):
        """
        Setup augmentation pipelines for training data
        Since dataset is very limited we rely on heavy augmentations to generate more data.
        Heavy augmentations are applied to the training set, while light augmentations are applied to the validation set.
        """
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.3),
            # A.RandomScale(scale_limit=0.2, p=0.5),
            # A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            # A.GridDistortion(p=0.3),
            # A.OpticalDistortion(p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.val_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def clean_labels(self, dataset_path, mapping):
        path = os.path.join(Path(dataset_path),'labels')
        for file in os.listdir(path):
            label = file.split('_')[0].strip()
            with open(os.path.join(path, file), 'r') as f:
                content = f.readlines()[0]

            new_label = mapping.get(label)
            parts = content.strip().split()
            parts[0] = str(new_label)
            final_str = ' '.join(parts) + '\n'

            with open(os.path.join(path,file),'w') as f:
                f.writelines(final_str)

    def apply_augmentation(self, image, annotations, split):
        """
        Apply augmentations to image and bounding box annotations
        
        Args:
            image: Input image (numpy array)
            annotations: YOLO format annotations [class, x_center, y_center, width, height]
            split: 'train' or 'val' to determine which augmentation pipeline to use
            
        Returns:
            augmented_image: Augmented image
            augmented_annotations: Transformed annotations in YOLO format
        """
        if split not in ['train', 'val']:
            return image, annotations
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                bboxes.append(list(map(float, parts[1:])))
                class_labels.append(cls)
        
        # Apply augmentation
        transform = self.train_transform if split == 'train' else self.val_transform
        
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            augmented_annotations = []
            for bbox, cls in zip(aug_bboxes, aug_labels):
                x_center, y_center, width, height = bbox
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Only keep valid boxes
                if width > 0 and height > 0:
                    augmented_annotations.append(f"{cls} {x_center} {y_center} {width} {height}")
            
            return aug_image, augmented_annotations
            
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}. Returning original image.")
            return image, annotations

    def process_dataset(self):
        """
        Process the entire dataset by splitting into train/val/test and applying augmentations.
        Reads images and labels from the dataset path and organizes them into YOLO structure.
        """
        print("Starting dataset processing...")
        
        # Setup output directories
        self.setup_directories()
        
        # Get all image files from dataset
        images_dir = self.dataset / 'images'
        labels_dir = self.dataset / 'labels'
        
        # Get all image files
        image_files = list(images_dir.glob('*.jpg'))
        
        print(f"Found {len(image_files)} images")
        
        # Split dataset: 70% train, 20% val, 10% test
        from sklearn.model_selection import train_test_split
        
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42) 
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Process each split
        splits = {
            'train': (train_files, 1),
            'val': (val_files, 1),
            'test': (test_files, 0)
        }
        
        for split_name, (files, num_augs) in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            for img_file in tqdm(files, desc=f"Processing {split_name}"):
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Warning: Could not read {img_file}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                label_file = labels_dir / img_file.name.replace('.jpg', '.txt').replace('.png', '.txt')
                
                if not label_file.exists():
                    print(f"Warning: Label file not found for {img_file.name}")
                    continue
                
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                
                if split_name == 'test':
                    filename = img_file.stem
                    self.save_sample(image, annotations, filename, split_name)
                else:
                    # Apply augmentations
                    for aug_idx in range(max(1, num_augs)):
                        aug_image, aug_annotations = self.apply_augmentation(
                            image.copy(), 
                            annotations.copy(), 
                            split_name
                        )
                        
                        # Create unique filename
                        if num_augs > 1:
                            filename = f"{img_file.stem}_aug{aug_idx}"
                        else:
                            filename = img_file.stem
                        
                        self.save_sample(aug_image, aug_annotations, filename, split_name)
        
        print("\nDataset processing completed!")
        print(f"Output directory: {self.processed_dir}")
        
    
    def save_sample(self, image, annotations, filename, split):
        """
        Save processed image and annotations to the output directory in YOLO format.
        
        Args:
            image: Image array (RGB format)
            annotations: List of annotation strings in YOLO format or single annotation lines
            filename: Base filename (without extension)
            split: Dataset split ('train', 'val', or 'test')
        """
        # Save image
        img_path = self.processed_dir / 'images' / split / f"{filename}.jpg"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), image_bgr)
        
        # Save annotations
        label_path = self.processed_dir / 'labels' / split / f"{filename}.txt"
        
        if isinstance(annotations, list):
            with open(label_path, 'w') as f:
                for ann in annotations:
                    ann_str = ann.strip()
                    if ann_str:  
                        f.write(ann_str + '\n')
        else:
            with open(label_path, 'w') as f:
                f.write(annotations.strip() + '\n')
        
    
    def create_yaml_config(self):
        """
        Create YOLO dataset configuration file
        This function generates a YAML configuration file for the YOLO dataset, specifying paths to training, validation, and test images,
        number of classes, and class names. The configuration file is essential for training YOLO models
        as it provides the necessary metadata about the dataset structure.
        """
        config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.num_classes,
            'names': [self.class_name_mapping[i] for i in range(self.num_classes)],
            'imgsz': self.imgsz,
        }
        
        yaml_path = self.processed_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        return yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for YOLO")  
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset directory")  
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")  
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    OUTPUT_DIR = args.output_dir

    processor = YOLODataProcessor(DATASET_PATH, OUTPUT_DIR)
    processor.process_dataset()
    yaml_path = processor.create_yaml_config()
    
    print("Dataset processing completed!")
    print(f"YAML config: {yaml_path}")
