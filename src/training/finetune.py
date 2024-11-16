from ultralytics import YOLO
from ..utils.patch import patch_ultralytics
from ..utils.read import read_augmentation_parameters
from ..utils.download import download_dataset
from ..utils.auth import setup_kaggle_auth
from ..utils.update import update_model

if __name__ == "__main__":
    # Set up Kaggle authentication
    if not setup_kaggle_auth():
        raise Exception("Failed to set up Kaggle authentication")

    # Download dataset if needed
    download_dataset()

    # Apply patches when running locally
    patch_ultralytics()

    # Paths
    model_name = "./models/yolo11x.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "finetuned-model"

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters("./runs/mosaic_erasing.txt")

    # Initialize the model
    model = YOLO(model_name)

    # Training parameters
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 600,
        "time": 0.5,
        "batch": 0.9,
        "cache": True,
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "optimizer": "auto",
        "seed": 42,
        "cos_lr": True,
        "fraction": 0.9,
        "multi_scale": True,
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
        model_dir="./runs/finetuned-model",
        title="Finetuned Vehicle Detection Model",
    )
    