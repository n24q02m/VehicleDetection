import os
import zipfile
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Đường dẫn
base_dir = "./data/soict-hackathon-2024_cvat-dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
train_images_dir = os.path.join(images_dir, "train")
train_labels_dir = os.path.join(labels_dir, "train")
valid_images_dir = os.path.join(images_dir, "valid")
valid_labels_dir = os.path.join(labels_dir, "valid")


# Giải nén các file zip
def extract_zip(zip_path):
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join("./data", zip_name)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Đang giải nén {zip_path} vào {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_dir)


# Đổi tên các file trong thư mục old
def rename_old_files(directory):
    print(f"Đang đổi tên các file trong {directory}...")
    processed_files = set()

    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Renaming"):
            file_path = os.path.join(root, file)
            if file_path not in processed_files and file.endswith(
                (".jpg", ".jpeg", ".png", ".txt")
            ):
                name, ext = os.path.splitext(file)
                if not name.endswith("_old"):
                    new_name = f"{name}_old{ext}"
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_name)
                    os.rename(old_path, new_path)
                    processed_files.add(file_path)


# Xử lý lại nhãn trong file txt
def process_label_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            label = int(parts[0])
            if label in [4, 5, 6, 7]:
                new_label = label - 4
                parts[0] = str(new_label)
                new_lines.append(" ".join(parts) + "\n")
            else:
                new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)


# Tạo file data.yaml
def create_yaml(train_files, valid_files):
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        f.write("path: ./\n")
        f.write("train: train.txt\n")
        f.write("valid: valid.txt\n\n")
        f.write("names:\n")
        f.write("  0: motorbike\n")
        f.write("  1: car\n")
        f.write("  2: bus\n")
        f.write("  3: truck\n")

    # Tạo file train.txt
    with open(os.path.join(base_dir, "train.txt"), "w") as f:
        for base_name in train_files:
            # Tìm file ảnh tương ứng với base_name trong thư mục train
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(train_images_dir, base_name + ext)):
                    f.write(f"images/train/{base_name}{ext}\n")
                    break

    # Tạo file valid.txt
    with open(os.path.join(base_dir, "valid.txt"), "w") as f:
        for base_name in valid_files:
            # Tìm file ảnh tương ứng với base_name trong thư mục valid
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(valid_images_dir, base_name + ext)):
                    f.write(f"images/valid/{base_name}{ext}\n")
                    break


# Xóa các thư mục không cần thiết
def cleanup_folders():
    print("Đang xóa các thư mục và file không cần thiết...")
    folders_to_remove = []

    for root, dirs, files in os.walk("./data"):
        for dir_name in dirs:
            if dir_name in ["train_20241023", "train_old_20241016"]:
                folders_to_remove.append(os.path.join(root, dir_name))

    for folder in tqdm(folders_to_remove, desc="Cleaning up folders"):
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Không thể xóa thư mục {folder}: {e}")


def zip_dataset():
    print("Đang nén dataset...")
    zip_name = os.path.basename(base_dir)
    zip_path = os.path.join(os.path.dirname(base_dir), f"{zip_name}.zip")

    # Tạo file zip mới
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Duyệt qua tất cả các file và thư mục trong base_dir
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                item_path = os.path.join(root, file)
                # Tạo đường dẫn tương đối từ tên folder con
                arcname = os.path.relpath(item_path, base_dir)
                zipf.write(item_path, arcname)

    print(f"Dataset đã được nén thành {zip_path}")


def post_process():
    print("Đang xử lý sau khi nén...")

    # Đổi tên folder gốc
    new_base_dir = "./data/soict-hackathon-2024_dataset"
    if os.path.exists(new_base_dir):
        shutil.rmtree(new_base_dir)
    os.rename(base_dir, new_base_dir)

    # Xóa các file txt không cần thiết
    for txt_file in ["train.txt", "valid.txt"]:
        txt_path = os.path.join(new_base_dir, txt_file)
        if os.path.exists(txt_path):
            os.remove(txt_path)

    # Đổi tên thư mục valid thành val
    old_valid_dir = os.path.join(new_base_dir, "images/valid")
    new_valid_dir = os.path.join(new_base_dir, "images/val")
    if os.path.exists(old_valid_dir):
        os.rename(old_valid_dir, new_valid_dir)

    old_valid_labels = os.path.join(new_base_dir, "labels/valid")
    new_valid_labels = os.path.join(new_base_dir, "labels/val")
    if os.path.exists(old_valid_labels):
        os.rename(old_valid_labels, new_valid_labels)

    # Sửa file data.yaml
    yaml_path = os.path.join(new_base_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            lines = f.readlines()

        with open(yaml_path, "w") as f:
            for line in lines:
                if line.startswith("train:"):
                    f.write("train: ./images/train\n")
                elif line.startswith("valid:"):
                    f.write("val: ./images/val\n")
                else:
                    f.write(line)


def remove_duplicates():
    print("Đang xử lý trùng lặp...")
    label_content_dict = {}
    duplicate_files = set()

    # Duyệt qua tất cả các file nhãn trong train_labels_dir
    for root, _, files in os.walk(train_labels_dir):
        for filename in tqdm(files, desc="Checking duplicates"):
            if filename.endswith(".txt"):
                label_path = os.path.join(root, filename)
                with open(label_path, 'r') as f:
                    content = f.read()
                if content in label_content_dict:
                    # Nếu trùng lặp, thêm cả nhãn và ảnh vào danh sách cần xóa
                    duplicate_files.add(label_path)
                    base_name = os.path.splitext(filename)[0]
                    # Tìm file ảnh tương ứng
                    for ext in [".jpg", ".jpeg", ".png"]:
                        image_path = os.path.join(train_images_dir, base_name + ext)
                        if os.path.exists(image_path):
                            duplicate_files.add(image_path)
                            break
                else:
                    label_content_dict[content] = label_path

    # Xóa các file trùng lặp
    for file_path in tqdm(duplicate_files, desc="Removing duplicates"):
        if os.path.exists(file_path):
            os.remove(file_path)

    print(f"Đã xóa {len(duplicate_files)} file trùng lặp.")


def main():
    if os.path.exists("./data/soict-hackathon-2024_dataset"):
        shutil.rmtree("./data/soict-hackathon-2024_dataset")

    if os.path.exists("./data/soict-hackathon-2024_cvat-dataset.zip"):
        os.remove("./data/soict-hackathon-2024_cvat-dataset.zip")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    extract_zip("./data/train_20241023.zip")
    extract_zip("./data/train_old_20241016.zip")

    rename_old_files("./data/train_old_20241016")

    print("Đang thu thập các file...")
    image_files = []
    label_files = []

    for root, _, files in os.walk("./data/train_20241023"):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
            elif file.endswith(".txt") and not file == "classes.txt":
                label_files.append(os.path.join(root, file))

    for root, _, files in os.walk("./data/train_old_20241016"):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
            elif file.endswith(".txt") and not file == "classes.txt":
                label_files.append(os.path.join(root, file))

    print("Đang xử lý và di chuyển các file...")
    for img_path in tqdm(image_files, desc="Processing images"):
        if os.path.exists(img_path):
            shutil.copy2(
                img_path, os.path.join(train_images_dir, os.path.basename(img_path))
            )

    for label_path in tqdm(label_files, desc="Processing labels"):
        if os.path.exists(label_path):
            new_label_path = os.path.join(
                train_labels_dir, os.path.basename(label_path)
            )
            shutil.copy2(label_path, new_label_path)
            process_label_file(new_label_path)

    # Gọi hàm loại bỏ trùng lặp trước khi tách tập train/val
    remove_duplicates()

    print("Đang tách tập validation...")
    all_files = [os.path.splitext(f)[0] for f in os.listdir(train_images_dir)]
    train_files, valid_files = train_test_split(
        all_files, test_size=0.2, random_state=42
    )

    print("Đang di chuyển các file validation...")
    for base_name in tqdm(valid_files, desc="Moving validation files"):
        img_extensions = [".jpg", ".jpeg", ".png"]
        for ext in img_extensions:
            src_img = os.path.join(train_images_dir, base_name + ext)
            if os.path.exists(src_img):
                shutil.move(
                    src_img, os.path.join(valid_images_dir, os.path.basename(src_img))
                )
                break

        src_label = os.path.join(train_labels_dir, base_name + ".txt")
        if os.path.exists(src_label):
            shutil.move(
                src_label, os.path.join(valid_labels_dir, os.path.basename(src_label))
            )

    create_yaml(train_files, valid_files)
    cleanup_folders()

    zip_dataset()
    post_process()

    print("Hoàn thành!")


if __name__ == "__main__":
    main()
