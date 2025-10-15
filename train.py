import torch
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import yaml
from config import *


class YOLOTrainer:
    def __init__(self, dataset_yaml, output_dir, model_size='n'):
        """
        Initialize YOLO trainer
        Args:
            dataset_yaml: Path to dataset YAML configuration
            output_dir: Directory to save training outputs
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        assert model_size in ['n', 's', 'm', 'l', 'x'], "Invalid model size. Choose from ['n', 's', 'm', 'l', 'x']"
        assert Path(dataset_yaml).is_file(), f"Dataset YAML file {dataset_yaml} does not exist"

        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.model_size = model_size
        self.device = [0,1]
        # print(f"Using device: {self.device}")
        
        # self.model = YOLO(f'yolo12{model_size}.pt')
        self.model = YOLO('yolo12-cls.yaml').load(f'yolo12{model_size}.pt') 
        self.num_workers = torch.multiprocessing.cpu_count()  


    def train(self):
        """
        Train YOLO segmentation model.
        This function sets up the training configuration and starts the training process.
        Returns:
            results: Training results including metrics like loss, mAP, etc.
        """        

        train_config = {
            'data': self.dataset_yaml,
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
        
        print("Starting YOLO segmentation training...")
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
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument('--dataset_yaml', type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save training outputs")

    args = parser.parse_args()

    trainer = YOLOTrainer(args.dataset_yaml, args.output_dir, MODEL_SIZE)

    train_results = trainer.train()

    val_results = trainer.validate()
    
    print("\nTraining completed!")
    print(f"Validation results: {val_results}")

if __name__ == "__main__":
    main()