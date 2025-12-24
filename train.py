import torch
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import yaml
from config import *


class YOLOTrainer:
    def __init__(self, dataset_path, output_dir, model_size='n', mode='bbox'):
        """
        Initialize YOLO trainer
        Args:
            dataset_path: Path to dataset YAML (for bbox) or directory (for cls)
            output_dir: Directory to save training outputs
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            mode: 'bbox' for detection or 'cls' for classification
        """
        assert model_size in ['n', 's', 'm', 'l', 'x'], "Invalid model size. Choose from ['n', 's', 'm', 'l', 'x']"
        
        self.mode = mode
        self.output_dir = output_dir
        self.model_size = model_size
        self.device = [0,1]
        
        # For bbox mode, expect YAML file
        if self.mode == 'bbox':
            assert Path(dataset_path).is_file(), f"Dataset YAML file {dataset_path} does not exist"
            self.dataset_path = dataset_path
            self.model = YOLO(f'yolo12{self.model_size}.pt')
        
        # For cls mode, expect directory path or YAML (we'll extract the directory from YAML)
        elif self.mode == 'cls':
            # If YAML file is provided, extract the dataset directory from it
            if Path(dataset_path).is_file() and dataset_path.endswith('.yaml'):
                with open(dataset_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Use the 'path' field from YAML as the dataset directory
                    self.dataset_path = config.get('path')
                    if not self.dataset_path or not Path(self.dataset_path).exists():
                        raise ValueError(f"Invalid 'path' in YAML: {self.dataset_path}")
                    print(f"Extracted dataset directory from YAML: {self.dataset_path}")
            # If directory is provided directly, use it
            elif Path(dataset_path).is_dir():
                self.dataset_path = dataset_path
            else:
                raise ValueError(f"For classification, provide either a dataset directory or YAML file with 'path' field")
            
            self.model = YOLO('yolo12-cls.yaml').load(f'yolo12{self.model_size}.pt')

        self.num_workers = torch.multiprocessing.cpu_count()


    def train(self):
        """
        Train YOLO model (detection or classification).
        This function sets up the training configuration and starts the training process.
        Returns:
            results: Training results including metrics like loss, mAP, etc.
        """        

        train_config = {
            'data': self.dataset_path,  # Changed from self.dataset_yaml
            'epochs': EPOCHS,
            'imgsz': IMGSZ,
            'batch': BATCH_SIZE,
            'device': self.device,
            'workers': self.num_workers,
            'save': True,
            'cache': True,
            'amp': True,
            'val': True,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            # 'box': 7,
            # 'cls': 1.0,   # Giving high emphasis on classification loss (Because the interpretability is affected more by classification than localization)
        }
        
        mode_name = "classification" if self.mode == 'cls' else "detection"
        print(f"Starting YOLO {mode_name} training...")
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            train_config['project'] = self.output_dir
            train_config['name'] = 'yolo_training' 
            print(f"Training results will be saved to: {self.output_dir}/yolo_training")

        print(f"Configuration: {train_config}")
        
        results = self.model.train(**train_config)
        
        return results

    def validate(self):
        """
        Validate the trained model
        This function runs validation on the trained model using the validation dataset specified in the YAML file.
        Returns:
            results: Validation results including metrics like mAP, precision, recall, etc.
        """
        print("Validating model...")
        results = self.model.val()
        return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model (detection or classification)")
    parser.add_argument('--data', type=str, required=True, 
                       help="Path to dataset YAML file (for bbox) or dataset directory/YAML (for cls)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save training outputs")
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help="YOLO model size")
    parser.add_argument('--mode', type=str, default='bbox', choices=['bbox', 'cls'], 
                       help="Training mode: 'bbox' for detection, 'cls' for classification")

    args = parser.parse_args()

    trainer = YOLOTrainer(args.data, args.output_dir, model_size=args.model_size, mode=args.mode)

    train_results = trainer.train()

    val_results = trainer.validate()
    
    print("\nTraining completed!")
    print(f"Validation results: {val_results}")

if __name__ == "__main__":
    main()