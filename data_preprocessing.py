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

        # self.clean_labels(self.dataset, self.label_class_mapping)
    
    def setup_directories(self):
        """
        Create YOLO dataset directory structure.
        As the dataset is limited we can use CPU to perform augmentations and store the images locally.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(self.processed_dir / 'images' / split, exist_ok=True)
            os.makedirs(self.processed_dir / 'labels' / split, exist_ok=True)


    def setup_augmentations(self,mode='box'):
        """
        Setup augmentation pipelines for training data
        """
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomCrop(height=self.imgsz, width=self.imgsz, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.3),
            # A.RandomScale(scale_limit=0.2, p=0.5),
            # A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            # A.GridDistortion(p=0.3),
            # A.OpticalDistortion(p=0.3),
        ]
        
        val_transform = [
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]

        if mode=='box':
            box_params = A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_visibility=0.3,  # Keep boxes with at least 30% visibility
            clip=True  # Clip bboxes to valid range
            )

            self.train_transform = A.Compose(train_transform, bbox_params=box_params)
            self.val_transform = A.Compose(val_transform, bbox_params=box_params)
        elif mode=='cls':
            self.train_transform = A.Compose(train_transform)
            self.val_transform = A.Compose(val_transform)

        
    # def clean_labels(self, dataset_path, mapping):
    #     path = os.path.join(Path(dataset_path),'labels')
    #     for file in os.listdir(path):
    #         label = file.split('_')[0].strip()
    #         with open(os.path.join(path, file), 'r') as f:
    #             content = f.readlines()[0]

    #         new_label = mapping.get(label)
    #         parts = content.strip().split()
    #         parts[0] = str(new_label)
    #         final_str = ' '.join(parts) + '\n'

    #         with open(os.path.join(path,file),'w') as f:
    #             f.writelines(final_str)

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
        
        self.setup_augmentations()

        bboxes = []
        class_labels = []
        
        # Small epsilon to avoid floating point precision issues
        epsilon = 1e-6
        
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                # Parse and clamp bbox coordinates BEFORE augmentation
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Clamp to valid range [0, 1] with epsilon buffer
                x_center = max(epsilon, min(1 - epsilon, x_center))
                y_center = max(epsilon, min(1 - epsilon, y_center))
                width = max(epsilon, min(1 - epsilon, width))
                height = max(epsilon, min(1 - epsilon, height))
                
                bboxes.append([x_center, y_center, width, height])
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
                
                # Clamp again after augmentation to ensure valid range
                x_center = max(epsilon, min(1 - epsilon, x_center))
                y_center = max(epsilon, min(1 - epsilon, y_center))
                width = max(epsilon, min(1 - epsilon, width))
                height = max(epsilon, min(1 - epsilon, height))
                
                # Only keep valid boxes
                if width > 0 and height > 0:
                    augmented_annotations.append(f"{cls} {x_center} {y_center} {width} {height}")
            
            return aug_image, augmented_annotations
            
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}. Returning original image.")
            return image, annotations
    
    def apply_augmentation_cls(self, image, split):
        """
        Apply augmentations to image for classification (no bounding boxes)
        
        Args:
            image: Input image (numpy array)
            split: 'train' or 'val' to determine which augmentation pipeline to use
            
        Returns:
            augmented_image: Augmented image
        """
        if split not in ['train', 'val']:
            return image
        
        self.setup_augmentations(mode='cls')

        
        # Apply augmentation
        transform = self.train_transform if split == 'train' else self.val_transform
        
        try:
            transformed = transform(image=image)
            aug_image = transformed['image']
            return aug_image
            
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}. Returning original image.")
            return image

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
    
    def process_classification_dataset(self, num_train_augs=2, num_val_augs=1):
        """
        Process dataset for YOLO classification model.
        Expects input directory structure:
            dataset_path/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image1.jpg
                    image2.jpg
        
        Args:
            num_train_augs: Number of augmented versions per training image
            num_val_augs: Number of augmented versions per validation image
        """
        print("Starting classification dataset processing...")
        
        # Get all class directories
        class_dirs = [d for d in self.dataset.iterdir() if d.is_dir()]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.dataset}")
        
        print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
        
        # Collect all images with their class labels
        all_data = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Get all image files
            image_files = list(class_dir.glob('*.jpg'))
            
            print(f"Class '{class_name}': {len(image_files)} images")
            
            for img_file in image_files:
                all_data.append({
                    'image_path': img_file,
                    'class_name': class_name
                })
        
        print(f"\nTotal images: {len(all_data)}")
        
        # Split data into train/val/test
        train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42, 
                                                   stratify=[d['class_name'] for d in all_data])
        val_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42,
                                                stratify=[d['class_name'] for d in temp_data])
        
        print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Create output directories for each split and class
        for split in ['train', 'val', 'test']:
            for class_dir in class_dirs:
                class_output_dir = self.processed_dir / split / class_dir.name
                os.makedirs(class_output_dir, exist_ok=True)
        
        # Process each split
        splits = {
            'train': (train_data, num_train_augs),
            'val': (val_data, num_val_augs),
            'test': (test_data, 0)  # No augmentation for test
        }
        
        for split_name, (data, num_augs) in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            for item in tqdm(data, desc=f"Processing {split_name}"):
                img_path = item['image_path']
                class_name = item['class_name']
                
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # For test set, save original image without augmentation
                if split_name == 'test':
                    self.save_classification_sample(image, class_name, img_path.stem, split_name)
                else:
                    # Save original image
                    self.save_classification_sample(image, class_name, img_path.stem, split_name)
                    
                    # Apply augmentations
                    for aug_idx in range(num_augs):
                        aug_image = self.apply_augmentation_cls(image.copy(), split_name)
                        filename = f"{img_path.stem}_aug{aug_idx}"
                        self.save_classification_sample(aug_image, class_name, filename, split_name)
        
        print("\nClassification dataset processing completed!")
        print(f"Output directory: {self.processed_dir}")
        
        # Print statistics
        for split in ['train', 'val', 'test']:
            for class_dir in class_dirs:
                class_output_dir = self.processed_dir / split / class_dir.name
                num_images = len(list(class_output_dir.glob('*.jpg')))
                print(f"{split}/{class_dir.name}: {num_images} images")
    
    def save_classification_sample(self, image, class_name, filename, split):
        """
        Save processed image for classification task
        
        Args:
            image: Image array (RGB format)
            class_name: Name of the class
            filename: Base filename (without extension)
            split: Dataset split ('train', 'val', or 'test')
        """
        # Create class directory if it doesn't exist
        class_dir = self.processed_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        
        # Save image
        img_path = class_dir / f"{filename}.jpg"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), image_bgr)
        
    
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
        
    
    def create_yaml_config(self, mode='box'):
        """
        Create YOLO dataset configuration file
        This function generates a YAML configuration file for the YOLO dataset, specifying paths to training, validation, and test images,
        number of classes, and class names. The configuration file is essential for training YOLO models
        as it provides the necessary metadata about the dataset structure.
        
        Args:
            mode: 'box' for detection or 'cls' for classification
        """
        if mode == 'cls':
            # For classification, just need path and class names
            config = {
                'path': str(self.processed_dir.absolute()),
                'train': 'train',
                'val': 'val',
                'test': 'test',
                'nc': self.num_classes,
                'names': [self.class_name_mapping[i] for i in range(self.num_classes)],
            }
        else:
            # For detection, specify images folder
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
    parser.add_argument('--mode',type=str, default='box', choices=['box', 'cls'], help="Operation mode: 'box' for bounding box, 'cls' for classification")
    parser.add_argument('--train_augs', type=int, default=1, help="Number of augmentations per training image (for classification mode)")
    parser.add_argument('--val_augs', type=int, default=1, help="Number of augmentations per validation image (for classification mode)")
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    OUTPUT_DIR = args.output_dir

    processor = YOLODataProcessor(DATASET_PATH, OUTPUT_DIR)

    if args.mode == 'box':
        processor.process_dataset()
        yaml_path = processor.create_yaml_config(mode='box')
    else:
        processor.process_classification_dataset(num_train_augs=args.train_augs, num_val_augs=args.val_augs)
        yaml_path = processor.create_yaml_config(mode='cls')
    
    print("Dataset processing completed!")
    print(f"YAML config: {yaml_path}")
