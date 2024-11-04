from ultralytics import YOLO

if __name__ == "__main__":
    model_name = "./runs/train-yolov8m-ghost-p2/weights/best.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "train-yolov8m-ghost-p2(2)"

    model = YOLO(model_name)

    model.train(
        data=f"{data_dir}/data.yaml",
        epochs=300,  # 300 cho train from scratch, 100 cho finetune
        time=1,
        patience=5,  # 30 cho train from scratch, 10 cho finetune
        batch=-1,  # -1 là tự động chọn batch size
        save_period=5,
        cache="disk",  # True nếu cache ram
        device=0,
        project=train_project,
        name=train_name,
        exist_ok=True,
        pretrained=True,  # True nếu finetune
        optimizer="SGD",  # AdamW với finetune
        cos_lr=True,
        fraction=1.0,  # 0.1 là sử dụng 10% dữ liệu
        multi_scale=True,
        augment=True,
        show=True,
        lr0=1e-2,  # 1e-3 đối với finetune
        weight_decay=5e-4,  # 1e-5 đối với finetune
    )
