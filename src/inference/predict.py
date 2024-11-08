import os
import zipfile
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch


def extract_zip(zip_path):
    extract_dir = os.path.splitext(zip_path)[0]
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_dir)

    return extract_dir


def predict_images(model_path, images_dir, output_file):
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

    # Open output file
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
                    )

                    # Get image dimensions from result
                    image_width = result.image_width
                    image_height = result.image_height

                    # Write results to file
                    for object_prediction in result.object_prediction_list:
                        # Get bounding box coordinates and confidence
                        x1, y1, x2, y2 = object_prediction.bbox.to_xyxy()
                        x_center = ((x1 + x2) / 2) / image_width
                        y_center = ((y1 + y2) / 2) / image_height
                        width = (x2 - x1) / image_width
                        height = (y2 - y1) / image_height
                        conf = object_prediction.score.value
                        cls = object_prediction.category.id

                        # Write in the required format
                        f.write(f"{file} {cls} {x_center} {y_center} {width} {height} {conf:.3f}\n")


def zip_output(output_file):
    print("Đang nén file kết quả...")
    with zipfile.ZipFile(f"{output_file}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Đã nén thành công: {output_file}.zip")


def main():
    # Paths
    test_zip = "./data/public test.zip"
    model_path = "./runs/better-train-yolov8m-ghost-p2/weights/best.pt"
    output_file = "./data/predict.txt"

    # Extract test images
    test_dir = extract_zip(test_zip)

    # Perform prediction and write results
    predict_images(model_path, test_dir, output_file)

    # Zip the output file
    zip_output(output_file)

    print("Hoàn thành!")


if __name__ == "__main__":
    main()
