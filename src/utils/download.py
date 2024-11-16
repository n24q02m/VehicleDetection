import os
import kaggle
from pathlib import Path


def download_dataset(
    dataset_name="n24q02m/augmented-vehicle-detection-dataset",
    dataset_dir="./data/soict-hackathon-2024_dataset",
):
    """
    Download dataset if not exists locally.

    Args:
        dataset_name (str): Kaggle dataset name in format username/dataset-slug
        dataset_dir (str): Local directory to save dataset
    """
    if not os.path.exists(dataset_dir):
        print(f"Downloading dataset {dataset_name}...")
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            kaggle.api.dataset_download_files(
                dataset_name, path=dataset_dir, unzip=True
            )
            print("Dataset downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    else:
        print("Dataset directory already exists. Skipping download.")
    return True


def download_model(
    model_name="n24q02m/finetuned-vehicle-detection-model",
    model_dir="./runs/finetune_yolo11x/weights",
):
    """
    Download pre-trained model if not exists locally.

    Args:
        model_name (str): Kaggle model name in format username/model-slug
        model_dir (str): Local directory to save model
    """
    if not os.path.exists(model_dir):
        print(f"Downloading model {model_name}...")
        os.makedirs(model_dir, exist_ok=True)
        try:
            kaggle.api.dataset_download_files(model_name, path=model_dir, unzip=True)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    else:
        print("Model directory already exists. Skipping download.")
    return True
