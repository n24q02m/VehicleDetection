import os
import zipfile
from ultralytics import YOLO
from tqdm import tqdm


def extract_zip(zip_path):
    extract_dir = os.path.splitext(zip_path)[0]
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_dir)

    return extract_dir


def predict_images(model_path, images_dir, output_file):
    # Tải model
    model = YOLO(model_path)

    # Tạo file predict.txt
    with open(output_file, "w") as f:
        # Duyệt qua tất cả các file trong thư mục
        for root, _, files in os.walk(images_dir):
            for file in tqdm(files, desc="Predicting"):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)

                    # Thực hiện predict
                    results = model.predict(image_path, conf=0.25)

                    # Ghi kết quả vào file
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Lấy tọa độ và confidence
                            x, y, w, h = box.xywhn[0].tolist()
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())

                            # Ghi theo định dạng yêu cầu
                            f.write(f"{file} {cls} {x} {y} {w} {h} {conf:.3f}\n")


def zip_output(output_file):
    print("Đang nén file kết quả...")
    with zipfile.ZipFile(f"{output_file}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Đã nén thành công: {output_file}.zip")


def main():
    # Các đường dẫn
    test_zip = "./data/public test.zip"
    model_path = "./src/training/runs/train/weights/best.pt"
    output_file = "./data/predict.txt"

    # Giải nén file test
    test_dir = extract_zip(test_zip)

    # Thực hiện predict và ghi kết quả
    predict_images(model_path, test_dir, output_file)

    # Nén file kết quả
    zip_output(output_file)

    print("Hoàn thành!")


if __name__ == "__main__":
    main()
