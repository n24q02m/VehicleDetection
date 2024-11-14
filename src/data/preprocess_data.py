import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def move_files_to_dataset(source_folder, target_images_folder, target_labels_folder):
    image_extensions = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                shutil.move(os.path.join(root, file), target_images_folder)
            elif file.lower().endswith(".txt"):
                shutil.move(os.path.join(root, file), target_labels_folder)
    print("Đã di chuyển các tệp vào thư mục dataset.")


def process_labels(labels_folder):
    for label_file in tqdm(os.listdir(labels_folder), desc="Processing labels"):
        label_path = os.path.join(labels_folder, label_file)
        new_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if 4 <= class_id <= 7:
                        class_id -= 4  # Chuyển 4->0, 5->1, 6->2, 7->3
                    new_line = " ".join([str(class_id)] + parts[1:])
                    new_lines.append(new_line)
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines))
    print("Đã xử lý lại nhãn.")


def create_background_images(images_folder, labels_folder):
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for image_file in tqdm(image_files, desc="Creating background images"):
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(
            labels_folder, os.path.splitext(image_file)[0] + ".txt"
        )
        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    mask[y1:y2, x1:x2] = 255
            inpainted_image = cv2.inpaint(
                image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA
            )
            bg_image_path = os.path.join(images_folder, f"bg_{image_file}")
            cv2.imwrite(bg_image_path, inpainted_image)
    print("Đã tạo các ảnh nền.")


def multiply_images(images_folder, labels_folder, multiplication_factor=2):
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("v")
    ]
    for image_file in tqdm(image_files, desc="Multiplying images"):
        for i in range(2, multiplication_factor + 1):
            src_image_path = os.path.join(images_folder, image_file)
            dest_image_path = os.path.join(images_folder, f"v{i}_{image_file}")
            shutil.copy(src_image_path, dest_image_path)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            src_label_path = os.path.join(labels_folder, label_file)
            dest_label_path = os.path.join(labels_folder, f"v{i}_{label_file}")
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
    print(f"Đã nhân số lượng ảnh lên {multiplication_factor} lần.")


def split_train_val(images_folder, labels_folder, val_ratio=0.2):
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    train_files, val_files = train_test_split(
        image_files, test_size=val_ratio, random_state=42
    )
    for val_file in tqdm(val_files, desc="Moving validation files"):
        image_path = os.path.join(images_folder, val_file)
        label_file = os.path.splitext(val_file)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_file)
        dest_image_path = os.path.join(images_folder.replace("train", "val"), val_file)
        dest_label_path = os.path.join(
            labels_folder.replace("train", "val"), label_file
        )
        shutil.move(image_path, dest_image_path)
        if os.path.exists(label_path):
            shutil.move(label_path, dest_label_path)
    print("Đã chia dữ liệu thành tập train và val.")


def main():
    extracted_folder = "./data/extracted_data"
    train_images_folder = "./data/soict-hackathon-2024_dataset/images/train"
    train_labels_folder = "./data/soict-hackathon-2024_dataset/labels/train"
    val_images_folder = "./data/soict-hackathon-2024_dataset/images/val"
    val_labels_folder = "./data/soict-hackathon-2024_dataset/labels/val"
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    move_files_to_dataset(extracted_folder, train_images_folder, train_labels_folder)
    shutil.rmtree(extracted_folder)
    process_labels(train_labels_folder)
    create_background_images(train_images_folder, train_labels_folder)
    multiply_images(train_images_folder, train_labels_folder, multiplication_factor=2)
    split_train_val(train_images_folder, train_labels_folder, val_ratio=0.2)

if __name__ == "__main__":
    main()
