import os
import logging
import yaml
import json
from datasets import Dataset, DatasetDict

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    logger.info(f"Config path being used (ABSOLUTE): {config_path}")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("YAML loaded successfully.")
    return config

def preprocess_dataset(config: dict):
    dataset_path = config["paths"]["dataset"]
    output_dir = config["paths"]["processed_data"]

    dataset_path = os.path.abspath(dataset_path)
    output_dir = os.path.abspath(output_dir)

    logger.info(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of dicts")

    dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict({"train": dataset})

    logger.info(f"Saving processed dataset to: {output_dir}")
    dataset_dict.save_to_disk(output_dir)
    logger.info("Dataset saved successfully.")

if __name__ == "__main__":
    try:
        config_file = "D:/221BCADA55/SJU/Projects/ML/ML Chatbot/config/settings.yaml"
        config = load_config(config_file)
        preprocess_dataset(config)
        logger.info("preprocess_dataset function COMPLETED successfully.")
        logger.info(f"Processed dataset should be saved to: {config['paths']['processed_data']}")
    except Exception as e:
        logger.error(f"Script failed: {e}")
