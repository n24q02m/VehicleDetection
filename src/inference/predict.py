import os
import cv2
import zipfile
from tqdm import tqdm
from ultralytics import YOLO
import torch
from pathlib import Path
from preprocess_image import preprocess_image

model_name = "better-train-yolov8m-ghost-p2"

def extract_zip(zip_path):
    extract_dir = os.path.splitext(zip_path)[0]
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_dir)

    return extract_dir


def predict_images(model_path, images_dir, output_file, labels_dir):
    # Load the model
    model = YOLO(model_path)
    
    predict_project = "./runs"
    predict_name = f"predict_{model_name}"
    
    # Load the model with quantization
    model.model.fuse()
    model.model.model = torch.quantization.quantize_dynamic(
        model.model.model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Create predictions file and labels directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Open output file for predictions
    with open(output_file, "w") as f:
        # Iterate over all image files
        for root, _, files in os.walk(images_dir):
            for file in tqdm(files, desc="Predicting"):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Không thể đọc file ảnh: {image_path}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
                    processed_image = preprocess_image(image)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  # Chuyển đổi từ RGB sang BGR

                    # Perform prediction using YOLO
                    results = model.predict(
                        source=processed_image,
                        conf=0.25,
                        iou=0.7,
                        device=0,
                        agnostic_nms=True,
                        retina_masks=True,
                        project=predict_project,
                        name=predict_name,
                    )

                    # Get image dimensions
                    image_width, image_height = results[0].orig_shape[1], results[0].orig_shape[0]

                    # Create label file for current image
                    base_name = os.path.splitext(file)[0]
                    txt_filepath = os.path.join(labels_dir, f"{base_name}.txt")

                    # Write results to both predict.txt and individual label files
                    with open(txt_filepath, "w") as txt_file:
                        for result in results[0].boxes:
                            x1, y1, x2, y2 = result.xyxy[0]
                            conf = result.conf[0]
                            cls = result.cls[0]
                            x_center = ((x1 + x2) / 2) / image_width
                            y_center = ((y1 + y2) / 2) / image_height
                            width = (x2 - x1) / image_width
                            height = (y2 - y1) / image_height

                            # Write to predict.txt
                            f.write(f"{file} {int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.3f}\n")
                            
                            # Write to individual label file
                            txt_file.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def zip_output(output_file):
    print("Đang nén file kết quả...")
    with zipfile.ZipFile(f"{output_file}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Đã nén thành công: {output_file}.zip")

def main():
    # Set up paths based on model directory
    
    model_path = f"./runs/{model_name}/weights/best.pt"
    model_dir = str(Path(model_path).parent.parent)  # Get model session directory

    test_zip = "./data/public test.zip"
    test_dir = "./data/public test"  # Define expected test directory
    output_file = os.path.join(model_dir, "predict.txt")
    labels_dir = os.path.join(model_dir, "labels")

    # Check if test directory already exists
    if not os.path.exists(test_dir):
        # Extract test images only if directory doesn't exist
        test_dir = extract_zip(test_zip)
        print("Đã giải nén xong thư mục test.")
    else:
        print("Đã tồn tại thư mục test, bỏ qua bước giải nén.")

    # Perform prediction and write results
    predict_images(model_path, test_dir, output_file, labels_dir)

    # Zip the output file
    zip_output(output_file)

    print("Hoàn thành!")

if __name__ == "__main__":
    main()