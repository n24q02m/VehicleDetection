import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultralytics import YOLO
from src.utils.patch import patch_ultralytics
from src.utils.read import read_augmentation_parameters
from src.utils.download import download_dataset, download_model
from src.utils.auth import setup_kaggle_auth
from src.utils.update import update_model


def main(train_mode="new", train_time=None, train_batch=-1):
    # Set up Kaggle authentication
    if not setup_kaggle_auth():
        raise Exception("Failed to set up Kaggle authentication")

    # Download dataset if needed
    download_dataset()

    # Download model from Kaggle
    if not download_model():
        raise Exception("Failed to download finetuned model from Kaggle")

    # Apply patches when running locally
    # patch_ultralytics()

    # Paths
    model_dir = Path("./models")
    best_model_path = model_dir / "finetuned_best.pt"
    initial_model_path = model_dir / "yolo11x.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"

    # Download yolo11x model
    yolo_model = YOLO(initial_model_path)

    # Add timestamp to train_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_name = f"finetuned-model_{timestamp}"

    # Determine model path based on train_mode
    if train_mode == "continue" and best_model_path.exists():
        model_name = str(best_model_path)
    else:
        model_name = initial_model_path

    # Verify model exists
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model not found at {model_name}.")

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters("./runs/mosaic_erasing.txt")

    # Initialize the model
    model = YOLO(model_name)

    # Training parameters
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 100,
        "time": train_time,
        "batch": train_batch,
        "imgsz": 480,
        "cache": "disk",
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "optimizer": "auto",
        "seed": 42,
        "cos_lr": True,
        "fraction": 1.0,
        "multi_scale": True,
        "half": True,
        "augment": True,
        "show": True,
        "label_smoothing": 0.1,
        **augmentation_params,
    }

    # Start training
    model.train(**train_params)

    # Update model on Kaggle
    update_model(
        model_name="n24q02m/finetuned-vehicle-detection-model",
        model_dir=str(
            Path(
                f"./runs/{train_name}/weights",
            ).absolute()
        ),
        title="Finetuned Vehicle Detection Model",
    )


if __name__ == "__main__":
    os.system("set KMP_DUPLICATE_LIB_OK=TRUE")
    os.system("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    main(train_mode="continue", time=8, batch=14)  # or "continue"
