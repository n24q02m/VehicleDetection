import os
import zipfile
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
from pathlib import Path


def extract_zip(zip_path):
    extract_dir = os.path.splitext(zip_path)[0]
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_dir)

    return extract_dir


def predict_images(model_path, images_dir, output_file, labels_dir):
    # Load the model with quantization
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0",
    )
    detection_model.model.fuse()
    detection_model.model.model = torch.quantization.quantize_dynamic(
        detection_model.model.model,
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

                    # Perform sliced prediction using SAHI
                    result = get_sliced_prediction(
                        image_path,
                        detection_model,
                        slice_height=512,
                        slice_width=512,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        postprocess_type="NMS",
                        postprocess_match_metric="IOU",
                        postprocess_match_threshold=0.2,
                    )

                    # Get image dimensions from result
                    image_width = result.image_width
                    image_height = result.image_height

                    # Create label file for current image
                    base_name = os.path.splitext(file)[0]
                    txt_filepath = os.path.join(labels_dir, f"{base_name}.txt")

                    # Write results to both predict.txt and individual label files
                    with open(txt_filepath, "w") as txt_file:
                        for object_prediction in result.object_prediction_list:
                            x1, y1, x2, y2 = object_prediction.bbox.to_xyxy()
                            x_center = ((x1 + x2) / 2) / image_width
                            y_center = ((y1 + y2) / 2) / image_height
                            width = (x2 - x1) / image_width
                            height = (y2 - y1) / image_height
                            conf = object_prediction.score.value
                            cls = object_prediction.category.id

                            # Write to predict.txt
                            f.write(f"{file} {cls} {x_center} {y_center} {width} {height} {conf:.3f}\n")
                            
                            # Write to individual label file
                            txt_file.write(f"{cls} {x_center} {y_center} {width} {height}\n")


def zip_output(output_file):
    print("Đang nén file kết quả...")
    with zipfile.ZipFile(f"{output_file}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Đã nén thành công: {output_file}.zip")


def main():
    # Set up paths based on model directory
    model_path = "./runs/better-train-yolov8m-ghost-p2/weights/best.pt"
    model_dir = str(Path(model_path).parent.parent)  # Get model session directory
    
    test_zip = "./data/public test.zip"
    test_dir = "./data/public test"  # Define expected test directory
    output_file = os.path.join(model_dir, "predict_sahi.txt")
    labels_dir = os.path.join(model_dir, "labels_sahi")

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
