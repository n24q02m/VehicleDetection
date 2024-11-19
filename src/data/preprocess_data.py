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
    """Create background images only for images that have bounding boxes."""
    # Get list of images that have corresponding label files
    label_files = {
        os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".txt")
    }
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.splitext(f)[0] in label_files
    ]

    # Process images in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    import threading

    thread_local = threading.local()

    def process_image(image_file):
        # Initialize thread-local OpenCV to avoid conflicts
        if not hasattr(thread_local, "cv2"):
            thread_local.cv2 = __import__("cv2")

        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(
            labels_folder, os.path.splitext(image_file)[0] + ".txt"
        )

        image = thread_local.cv2.imread(image_path)
        if image is None:
            return

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                x1 = max(0, int((x_center - width / 2) * w))
                y1 = max(0, int((y_center - height / 2) * h))
                x2 = min(w, int((x_center + width / 2) * w))
                y2 = min(h, int((y_center + height / 2) * h))
                mask[y1:y2, x1:x2] = 255

        inpainted_image = thread_local.cv2.inpaint(
            image, mask, inpaintRadius=3, flags=thread_local.cv2.INPAINT_TELEA
        )
        bg_image_path = os.path.join(images_folder, f"bg_{image_file}")
        thread_local.cv2.imwrite(bg_image_path, inpainted_image)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(
            tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="Creating background images",
            )
        )

    print("Đã tạo các ảnh nền cho ảnh có bbox.")


def multiply_images(images_folder, labels_folder, multiplication_factor=2):
    """Multiply images and labels using parallel processing."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Get list of original images (exclude previously multiplied ones)
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("v")
    ]

    def process_multiplication(args):
        image_file, version = args
        try:
            # Copy image
            src_image_path = os.path.join(images_folder, image_file)
            dest_image_path = os.path.join(images_folder, f"v{version}_{image_file}")
            shutil.copy2(src_image_path, dest_image_path)

            # Copy corresponding label if exists
            label_file = os.path.splitext(image_file)[0] + ".txt"
            src_label_path = os.path.join(labels_folder, label_file)
            if os.path.exists(src_label_path):
                dest_label_path = os.path.join(
                    labels_folder, f"v{version}_{label_file}"
                )
                shutil.copy2(src_label_path, dest_label_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Create tasks list
    tasks = [
        (img, ver) for img in image_files for ver in range(2, multiplication_factor + 1)
    ]

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(
            tqdm(
                executor.map(process_multiplication, tasks),
                total=len(tasks),
                desc="Multiplying images",
            )
        )

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
    # multiply_images(train_images_folder, train_labels_folder, multiplication_factor=2)
    split_train_val(train_images_folder, train_labels_folder, val_ratio=0.1)


if __name__ == "__main__":
    main()
