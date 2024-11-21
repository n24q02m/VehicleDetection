import os
import sys
import random
import zipfile
from datetime import datetime
from tqdm import tqdm
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.download import download_model


def setup_test_dir(test_dir="./data/public_test"):
    """Setup test directory by downloading from Kaggle if not exists."""
    if not os.path.exists(test_dir):
        print("Tải xuống và giải nén dữ liệu test...")
        try:
            import kaggle

            # Tạo thư mục public test trước
            os.makedirs(test_dir, exist_ok=True)

            # Tải xuống và giải nén trực tiếp vào thư mục public test
            kaggle.api.dataset_download_files(
                "n24q02m/public-test-vehicle-detection-data",
                path=test_dir,
                unzip=True,
                quiet=False,
            )
            print("Đã tải xuống và giải nén xong thư mục test.")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu test: {e}")
            return None
    else:
        print("Đã tồn tại thư mục test, bỏ qua bước tải xuống.")
    return test_dir


def preprocess_image(image, enhance=True):
    """
    Tiền xử lý hình ảnh để cải thiện chất lượng trước khi dự đoán.

    Args:
        image: Ảnh đầu vào (numpy array)
        enhance: Có áp dụng cải thiện chất lượng hay không

    Returns:
        Ảnh đã được xử lý
    """
    if not enhance:
        return image

    # Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Cân bằng histogram kênh L (độ sáng)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Ghép các kênh lại
    lab = cv2.merge([l, a, b])

    # Chuyển lại không gian màu BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Giảm nhiễu
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)

    # Tăng độ sắc nét
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def draw_predictions(image_path, label_path, output_path):
    """Vẽ bbox lên ảnh dựa trên file nhãn."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Class {int(cls)}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imwrite(output_path, image)


def create_preview(test_dir, labels_dir, preview_dir, num_samples=10):
    """Tạo ảnh preview với nhãn được vẽ lên."""
    print("Tạo ảnh preview...")
    os.makedirs(preview_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))

    for file in selected_files:
        image_path = os.path.join(test_dir, file)
        label_path = os.path.join(labels_dir, f"{os.path.splitext(file)[0]}.txt")
        output_path = os.path.join(preview_dir, f"preview_{file}")

        if os.path.exists(label_path):
            draw_predictions(image_path, label_path, output_path)


def predict_images(model_path, images_dir, output_dir, use_sahi=False, enhance=False):
    """Predict images using either YOLO or SAHI with optional image enhancement."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predict.txt")

    if use_sahi:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.25,
            device="cuda:0",
        )
    else:
        detection_model = YOLO(model_path)

    # Apply model optimization
    detection_model.model.fuse()
    detection_model.model.model = torch.quantization.quantize_dynamic(
        detection_model.model.model, {torch.nn.Linear}, dtype=torch.qint8
    )

    with open(output_file, "w") as f:
        for root, _, files in os.walk(images_dir):
            for file in tqdm(files, desc="Predicting"):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    txt_filepath = os.path.join(labels_dir, f"{base_name}.txt")

                    # Đọc và tiền xử lý ảnh
                    image = cv2.imread(image_path)
                    processed_image = preprocess_image(image, enhance)

                    if use_sahi:
                        result = get_sliced_prediction(
                            processed_image,
                            detection_model,
                            slice_height=512,
                            slice_width=512,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            postprocess_type="NMS",
                            postprocess_match_metric="IOU",
                            postprocess_match_threshold=0.2,
                        )
                        image_width = result.image_width
                        image_height = result.image_height

                        with open(txt_filepath, "w") as txt_file:
                            for pred in result.object_prediction_list:
                                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                                x_center = ((x1 + x2) / 2) / image_width
                                y_center = ((y1 + y2) / 2) / image_height
                                width = (x2 - x1) / image_width
                                height = (y2 - y1) / image_height
                                conf = pred.score.value
                                cls = pred.category.id

                                f.write(
                                    f"{file} {cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.3f}\n"
                                )
                                txt_file.write(
                                    f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                )
                    else:
                        results = detection_model.predict(
                            source=processed_image,
                            conf=0.25,
                            iou=0.7,
                            device=0,
                            agnostic_nms=True,
                            retina_masks=True,
                        )
                        image_width, image_height = (
                            results[0].orig_shape[1],
                            results[0].orig_shape[0],
                        )

                        with open(txt_filepath, "w") as txt_file:
                            for result in results[0].boxes:
                                x1, y1, x2, y2 = result.xyxy[0]
                                conf = result.conf[0]
                                cls = result.cls[0]
                                x_center = ((x1 + x2) / 2) / image_width
                                y_center = ((y1 + y2) / 2) / image_height
                                width = (x2 - x1) / image_width
                                height = (y2 - y1) / image_height

                                f.write(
                                    f"{file} {int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.3f}\n"
                                )
                                txt_file.write(
                                    f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                )

    # Create preview images
    preview_dir = os.path.join(output_dir, "preview")
    create_preview(images_dir, labels_dir, preview_dir)

    # Zip predict.txt
    print("Đang nén file kết quả...")
    with zipfile.ZipFile(f"{output_file}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Đã nén thành công: {output_file}.zip")

    return output_dir


def main(use_sahi=False, enhance=False):
    # Download model if needed
    download_model(
        model_name="n24q02m/final-vehicle-detection-model",
        model_dir="./models",
        best_model_filename="final_best.pt",
        last_model_filename="final_last.pt",
    )

    # Setup paths
    model_dir = "./models"
    best_model_path = os.path.join(model_dir, "final_best.pt")
    test_dir = setup_test_dir()
    if test_dir is None:
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./runs", f"predict_{timestamp}")

    # Perform prediction
    predict_images(best_model_path, test_dir, output_dir, use_sahi, enhance)
    print("Hoàn thành!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sahi", action="store_true", help="Use SAHI for prediction")
    parser.add_argument(
        "--enhance", action="store_true", help="Apply image enhancement"
    )
    args = parser.parse_args()

    main(use_sahi=args.sahi, enhance=args.enhance)
