import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultralytics import YOLO
from src.training.distillation import DistillationTrainer
from src.utils.patch import patch_ultralytics
from src.utils.read import read_augmentation_parameters
from src.utils.download import download_dataset, download_model
from src.utils.auth import setup_kaggle_auth
from src.utils.update import update_model


def main(train_mode="new"):
    # Set up Kaggle authentication
    if not setup_kaggle_auth():
        raise Exception("Failed to set up Kaggle authentication")

    # Download data and model if needed
    download_dataset()
    if not download_model():
        raise Exception("Failed to download teacher model")

    # Verify teacher model exists
    teacher_model_path = "./model/finetuned_best.pt"
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(
            f"Teacher model not found at {teacher_model_path}. "
            "Please ensure finetune.py was run first or the model was downloaded correctly."
        )

    # Apply patches when running locally
    # patch_ultralytics()

    # Paths
    model_dir = Path("./models")
    best_model_path = model_dir / "final_best.pt"
    initial_model_path = model_dir / "yolov8m-adsc.yaml"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"

    # Add timestamp to train_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_name = f"final-model_{timestamp}"

    # Determine model path based on train_mode
    if train_mode == "continue":
        if not download_model(
            model_name="n24q02m/final-vehicle-detection-model",
            best_model_filename="final_best.pt",
        ):
            raise Exception("Failed to download final model from Kaggle")
        model_name = str(best_model_path)
    else:
        model_name = initial_model_path

    # Verify model exists
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model not found at {model_name}.")

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters("./runs/mosaic_erasing.txt")

    # Initialize the student model
    model = YOLO(model_name)

    # Training parameters
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 100,
        "time": 4,
        "batch": 0.8,
        "imgsz": 640,
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

    # Start training with the custom distillation trainer
    model.train(trainer=DistillationTrainer, **train_params)

    # Update model on Kaggle
    update_model(
        model_name="n24q02m/final-vehicle-detection-model",
        model_dir=str(
            Path(
                f"./runs/{train_name}/weights",
            ).absolute()
        ),
        title="Final Vehicle Detection Model",
    )


if __name__ == "__main__":
    main(train_mode="new")  # Hoặc "continue"
