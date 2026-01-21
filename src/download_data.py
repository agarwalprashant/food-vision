"""
Downloads and prepares the pizza_steak_sushi dataset.

This script downloads the dataset from GitHub and extracts it
to the data directory.
"""
import sys
import yaml
from pathlib import Path
import logging

#logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('download_data_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Add parent directory to path so we can import from going_modular
sys.path.append(str(Path(__file__).resolve().parents[1]))

from going_modular import helper_functions

def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml."""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise    


def main():
    """Download the pizza_steak_sushi dataset."""
    try:
        # Load parameters
        params = load_params('params.yaml')
        
        # Define the data source and destination
        source_url = params['data_source']
        destination = params['data_dir']
        logger.debug('Source URL: %s, Destination: %s', source_url, destination)
        
        # Download and extract the data
        image_path = helper_functions.download_data(
            source=source_url,
            destination=destination
        )
        
        logger.debug('Data downloaded to: %s', image_path)
        
        # Verify train and test directories exist
        train_dir = image_path / "train"
        test_dir = image_path / "test"
        
        if train_dir.exists() and test_dir.exists():
            logger.debug('Train directory: %s', train_dir)
            logger.debug('Test directory: %s', test_dir)
            logger.info('Data download completed successfully!')
        else:
            logger.error('Train or test directories not found!')
            sys.exit(1)
    except Exception as e:
        logger.error('Failed to complete the download data process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error('Failed to complete the download data process: %s', e)
        print(f"Error: {e}")

