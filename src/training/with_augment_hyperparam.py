from ultralytics import YOLO


def read_augmentation_parameters(file_path):
    params = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key_value = line.strip().split(maxsplit=1)
                if len(key_value) == 2:
                    key, value = key_value
                    # Chuyển đổi giá trị thành float hoặc tuple nếu cần
                    try:
                        if "(" in value and ")" in value:
                            value = eval(value)
                        else:
                            value = float(value)
                    except ValueError:
                        pass  # Giữ nguyên chuỗi nếu không chuyển đổi được
                    params[key] = value
    return params


if __name__ == "__main__":
    # Đường dẫn
    model_path = "./runs/train-yolov8m-ghost-p2(2)/weights/best.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "train-yolov8m-ghost-p2(3)"

    # Đọc các tham số tăng cường từ tệp
    augmentation_params = read_augmentation_parameters(
        "./runs/augmentation-hyperparameter.txt"
    )

    # Khởi tạo mô hình
    model = YOLO(model_path)

    # Thiết lập các tham số huấn luyện
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 100,
        "time": 4,
        "patience": 5,
        "batch": -1,
        "save_period": 5,
        "cache": "disk",
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "cos_lr": True,
        "multi_scale": True,
        "augment": True,
        "show": True,
        "lr0": 1e-2,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
    }

    # Thêm các tham số tăng cường vào tham số huấn luyện
    train_params.update(augmentation_params)

    # Huấn luyện mô hình với các tham số đã thiết lập
    model.train(**train_params)
