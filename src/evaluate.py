"""
Evaluation script for the trained food classification model.

This script loads the trained model, evaluates it on the test set,
and generates loss curves and metrics.
"""
import sys
import json
import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from torchvision import transforms
import logging

# logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('evaluate_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Add parent directory to path so we can import from going_modular
sys.path.append(str(Path(__file__).resolve().parents[1]))

from going_modular import data_setup, engine


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


def plot_loss_curves(results, save_path):
    """
    Plots and saves training curves.
    
    Args:
        results: Dictionary containing train_loss, train_acc, test_loss, test_acc
        save_path: Path to save the figure
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.debug('Loss curves saved to: %s', save_path)


def main():
    """Main evaluation function."""
    try:
        # Load parameters
        params = load_params()
        
        # Setup device
        device = get_device()
        logger.debug('Using device: %s', device)
        
        # Setup directories
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / "data" / "pizza_steak_sushi"
        test_dir = data_path / "test"
        model_dir = project_root / "models"
        reports_dir = project_root / "reports"
        figures_dir = reports_dir / "figures"
        
        # Create output directories if they don't exist
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load class names
        class_names_path = model_dir / "class_names.json"
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        
        logger.debug('Class names: %s', class_names)
        
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
        
        # Create test data loader
        batch_size = params.get("batch_size", 32)
        from torchvision import datasets
        from torch.utils.data import DataLoader
        
        test_data = datasets.ImageFolder(test_dir, transform=manual_transforms)
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for evaluation to avoid potential issues
            pin_memory=True,
        )
        
        # Create model architecture
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(class_names))
        )
        
        # Load trained weights
        model_path = model_dir / f"{params.get('model_name', 'efficientnet_b0')}_baseline.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.debug('Model loaded from: %s', model_path)
        
        # Evaluate model on test set
        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc = engine.test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        logger.debug('Test Loss: %.4f', test_loss)
        logger.debug('Test Accuracy: %.4f', test_acc)
        
        # Load training metrics and plot loss curves
        training_metrics_path = reports_dir / "training_metrics.json"
        if training_metrics_path.exists():
            with open(training_metrics_path, "r") as f:
                training_metrics = json.load(f)
            
            # Plot and save loss curves
            results = {
                "train_loss": training_metrics["train_loss"],
                "train_acc": training_metrics["train_acc"],
                "test_loss": training_metrics["test_loss"],
                "test_acc": training_metrics["test_acc"]
            }
            plot_loss_curves(results, figures_dir / "loss_curves.png")
        
        # Save evaluation metrics
        eval_metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "num_test_samples": len(test_data),
            "num_classes": len(class_names),
            "class_names": class_names
        }
        
        eval_metrics_path = reports_dir / "evaluation_metrics.json"
        with open(eval_metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.debug('Evaluation metrics saved to: %s', eval_metrics_path)
        
        logger.info('Evaluation completed successfully!')
    except Exception as e:
        logger.error('Failed to complete the evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error('Failed to complete the evaluation process: %s', e)
        print(f"Error: {e}")
