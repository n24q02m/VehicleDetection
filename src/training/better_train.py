from ultralytics import YOLO


def read_augmentation_parameters(file_path):
    params = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key_value = line.strip().split(maxsplit=1)
                if len(key_value) == 2:
                    key, value = key_value
                    try:
                        if "(" in value and ")" in value:
                            value = eval(value)
                        else:
                            value = float(value)
                    except ValueError:
                        pass
                    params[key] = value
    return params


if __name__ == "__main__":
    # Paths
    model_name = "yolov8m-ghost-p2.yaml"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "better-train-yolov8m-ghost-p2"

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters(
        "./runs/augmentation-hyperparameter.txt"
    )

    # Extract only HSV parameters
    hsv_params = {
        k: augmentation_params[k]
        for k in ["hsv_h", "hsv_s", "hsv_v"]
        if k in augmentation_params
    }

    # Initialize the model
    model = YOLO(model_name)

    # Base training parameters
    base_train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 600,
        "batch": -1,
        "cache": "disk",
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "optimizer": "auto",
        "seed": 42,
        "cos_lr": True,
        "multi_scale": True,
        "augment": True,
        "show": True,
        "label_smoothing": 0.1,
        **hsv_params,
    }

    # List of training times
    training_times = [12]  # Times in hours

    for idx, training_time in enumerate(training_times):
        print(f"Bắt đầu training lần {idx + 1} với thời gian {training_time} tiếng...")
        train_params = base_train_params.copy()
        train_params["time"] = training_time

        # Start training
        model.train(**train_params)

        # After training, if not the last run, prompt to continue and load best model
        if idx < len(training_times) - 1:
            input("Nhấn Enter để tiếp tục đến lần training tiếp theo...")
            best_weights_path = model.trainer.best
            model = YOLO(best_weights_path)
