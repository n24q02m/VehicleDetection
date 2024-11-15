from ultralytics import YOLO
from train import read_augmentation_parameters, patch_ultralytics


if __name__ == "__main__":
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
        "time": 8,
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
