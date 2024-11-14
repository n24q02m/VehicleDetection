import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm


def get_image_paths(data_folder):
    image_paths = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_label_paths(data_folder):
    label_paths = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(".txt"):
                label_paths.append(os.path.join(root, file))
    return label_paths


def explore_images_combined(image_paths_dict, output_dir):
    """
    Vẽ biểu đồ số lượng ảnh cho nhiều tập dữ liệu trên cùng một biểu đồ.

    Args:
        image_paths_dict (dict): Từ điển với key là tên tập dữ liệu và value là danh sách đường dẫn ảnh.
        output_dir (str): Thư mục để lưu biểu đồ.
    """
    counts = {title: len(paths) for title, paths in image_paths_dict.items()}
    print("Số lượng hình ảnh:")
    for title, count in counts.items():
        print(f"{title}: {count} hình ảnh")

    # Vẽ biểu đồ
    plt.figure()
    plt.bar(counts.keys(), counts.values(), color=["blue", "green", "red"])
    plt.ylabel("Số lượng ảnh")
    plt.title("Số lượng ảnh trong các tập dữ liệu")
    plt.savefig(os.path.join(output_dir, "image_counts_combined.png"))
    plt.close()


def explore_labels(label_paths, title, output_dir):
    class_counts = {}
    total_labels = 0
    for label_path in tqdm(label_paths, desc=f"Processing {title} labels"):
        with open(label_path, "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_labels += 1
    print(f"{title}: {total_labels} nhãn")
    print(f"Phân bố nhãn: {class_counts}")
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    plt.figure()
    plt.bar(classes, counts)
    plt.title(f"Phân bố nhãn cho {title}")
    plt.xlabel("Class ID")
    plt.ylabel("Số lượng nhãn")
    plt.savefig(
        os.path.join(output_dir, f"{title.replace(' ', '_')}_label_distribution.png")
    )
    plt.close()
    return total_labels, class_counts


def save_random_images_with_boxes(
    image_paths, labels_folder, output_folder, num_images=3
):
    os.makedirs(output_folder, exist_ok=True)
    random_images = random.sample(image_paths, num_images)
    for image_path in random_images:
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        label_path = os.path.join(
            labels_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        )
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
    print(f"Đã lưu {num_images} hình ảnh với nhãn vào {output_folder}")


def analyze_box_dimensions(label_paths, output_dir):
    widths = []
    heights = []
    for label_path in tqdm(label_paths, desc="Analyzing box dimensions"):
        with open(label_path, "r") as f:
            for line in f:
                _, _, _, width, height = map(float, line.strip().split())
                widths.append(width)
                heights.append(height)
    plt.figure()
    plt.hist(widths, bins=50, alpha=0.5, label="Width")
    plt.hist(heights, bins=50, alpha=0.5, label="Height")
    plt.legend()
    plt.title("Distribution of Box Widths and Heights")
    plt.xlabel("Normalized Size")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "box_dimensions.png"))
    plt.close()
    print("Đã lưu biểu đồ phân bố kích thước bounding boxes.")


def analyze_box_heatmap(image_paths, labels_folder, output_dir):
    heatmap = None
    for image_path in tqdm(image_paths, desc="Creating heatmap"):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        label_path = os.path.join(
            labels_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        )
        if os.path.exists(label_path):
            mask = np.zeros((h, w))
            with open(label_path, "r") as f:
                for line in f:
                    _, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    mask[y1:y2, x1:x2] += 1
            if heatmap is None:
                heatmap = mask
            else:
                heatmap += mask
    plt.figure()
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title("Heatmap of Bounding Boxes")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "box_heatmap.png"))
    plt.close()
    print("Đã lưu biểu đồ heatmap của bounding boxes.")


def analyze_image_colors(image_paths, output_dir):
    brightness_values = []
    contrast_values = []
    saturation_values = []
    for image_path in tqdm(image_paths, desc="Analyzing image colors"):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness_values.append(np.mean(hsv[:, :, 2]))
        saturation_values.append(np.mean(hsv[:, :, 1]))
        contrast_values.append(np.std(hsv[:, :, 2]))
    plt.figure()
    plt.hist(brightness_values, bins=50)
    plt.title("Brightness Distribution")
    plt.savefig(os.path.join(output_dir, "brightness_distribution.png"))
    plt.close()
    plt.figure()
    plt.hist(contrast_values, bins=50)
    plt.title("Contrast Distribution")
    plt.savefig(os.path.join(output_dir, "contrast_distribution.png"))
    plt.close()
    plt.figure()
    plt.hist(saturation_values, bins=50)
    plt.title("Saturation Distribution")
    plt.savefig(os.path.join(output_dir, "saturation_distribution.png"))
    plt.close()
    print("Đã lưu biểu đồ phân bố màu sắc.")


def main():
    os.makedirs("./runs/explore_dataset", exist_ok=True)
    output_dir = "./runs/explore_dataset"

    extracted_folder = "./data/extracted_data"
    daytime_folder = os.path.join(extracted_folder, "daytime")
    nighttime_folder = os.path.join(extracted_folder, "nighttime")

    daytime_images = get_image_paths(daytime_folder)
    nighttime_images = get_image_paths(nighttime_folder)
    total_images = daytime_images + nighttime_images

    daytime_labels = get_label_paths(daytime_folder)
    nighttime_labels = get_label_paths(nighttime_folder)
    total_labels = daytime_labels + nighttime_labels

    # Tạo dictionary chứa các tập dữ liệu
    image_paths_dict = {
        "Tổng số": total_images,
        "Ban ngày": daytime_images,
        "Ban đêm": nighttime_images
    }

    # Gọi hàm explore_images_combined
    explore_images_combined(image_paths_dict, output_dir)

    explore_labels(total_labels, "Tổng số nhãn", output_dir)
    explore_labels(daytime_labels, "Nhãn ban ngày", output_dir)
    explore_labels(nighttime_labels, "Nhãn ban đêm", output_dir)

    save_random_images_with_boxes(
        daytime_images, daytime_folder, os.path.join(output_dir, "daytime_samples")
    )
    save_random_images_with_boxes(
        nighttime_images,
        nighttime_folder,
        os.path.join(output_dir, "nighttime_samples"),
    )

    analyze_box_dimensions(total_labels, output_dir)
    analyze_box_heatmap(total_images, extracted_folder, output_dir)
    analyze_image_colors(total_images, output_dir)

if __name__ == "__main__":
    main()
