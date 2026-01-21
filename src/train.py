"""
Training script for the EfficientNet-B0 food classification model.

This script trains an EfficientNet-B0 model on the pizza_steak_sushi dataset
using transfer learning with frozen base layers.
"""
import sys
import json
import yaml
import torch
import torchvision
from pathlib import Path
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
import logging

# logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('train_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Add parent directory to path so we can import from going_modular
sys.path.append(str(Path(__file__).resolve().parents[1]))

from going_modular import data_setup, engine, helper_functions


def load_params():
    """Load parameters from params.yaml."""
    try:
        params_path = Path(__file__).resolve().parents[1] / "params.yaml"
        if not params_path.exists():
            logger.error('params.yaml not found: %s', params_path)
            raise FileNotFoundError(f"params.yaml not found at {params_path}")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading params: %s', e)
        raise


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    """Main training function."""
    try:
        # Load parameters
        params = load_params()
        
        # Set seeds for reproducibility
        helper_functions.set_seeds(seed=params.get("seed", 42))
        
        # Setup device
        device = get_device()
        logger.debug('Using device: %s', device)
        
        # Setup directories
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / "data" / "pizza_steak_sushi"
        train_dir = data_path / "train"
        test_dir = data_path / "test"
        model_dir = project_root / "models"
        reports_dir = project_root / "reports"
        
        # Create output directories if they don't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        manual_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        # Create data loaders
        batch_size = params.get("batch_size", 32)
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=manual_transforms,
            batch_size=batch_size
        )
        
        logger.debug('Number of classes: %s', len(class_names))
        logger.debug('Class names: %s', class_names)
        
        # Create model
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        
        # Freeze base layers
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Update classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(class_names))
        ).to(device)
        
        # Define loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        learning_rate = params.get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        epochs = params.get("epochs", 5)
        logger.info('Starting training for %s epochs...', epochs)
        
        start_time = timer()
        
        results = engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device
        )
        
        end_time = timer()
        total_time = end_time - start_time
        logger.debug('Total training time: %.3f seconds', total_time)
        
        # Save the trained model
        model_path = model_dir / f"{params.get('model_name', 'efficientnet_b0')}_baseline.pth"
        torch.save(model.state_dict(), model_path)
        logger.debug('Model saved to: %s', model_path)
        
        # Save class names for inference
        class_names_path = model_dir / "class_names.json"
        with open(class_names_path, "w") as f:
            json.dump(class_names, f)
        logger.debug('Class names saved to: %s', class_names_path)
        
        # Save training metrics
        metrics = {
            "train_loss": results["train_loss"],
            "train_acc": results["train_acc"],
            "test_loss": results["test_loss"],
            "test_acc": results["test_acc"],
            "final_train_loss": results["train_loss"][-1],
            "final_train_acc": results["train_acc"][-1],
            "final_test_loss": results["test_loss"][-1],
            "final_test_acc": results["test_acc"][-1],
            "total_training_time_seconds": total_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
        metrics_path = reports_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.debug('Training metrics saved to: %s', metrics_path)
        
        logger.info('Training completed successfully!')
    except Exception as e:
        logger.error('Failed to complete the training process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error('Failed to complete the training process: %s', e)
        print(f"Error: {e}")
