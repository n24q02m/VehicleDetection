import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


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
    """Explore labels using parallel processing."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    thread_local = threading.local()
    class_counts = {}
    total_labels = 0

    def process_label_file(label_path):
        counts = {}
        num_labels = 0
        with open(label_path, "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counts[class_id] = counts.get(class_id, 0) + 1
                num_labels += 1
        return counts, num_labels

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(
            tqdm(
                executor.map(process_label_file, label_paths),
                total=len(label_paths),
                desc=f"Processing {title} labels",
            )
        )

    # Aggregate results
    class_counts = {}
    total_labels = 0
    for counts, num_labels in results:
        total_labels += num_labels
        for class_id, count in counts.items():
            class_counts[class_id] = class_counts.get(class_id, 0) + count

    print(f"{title}: {total_labels} nhãn")
    print(f"Phân bố nhãn: {class_counts}")

    # Plot distribution
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


def analyze_bbox_heatmap(label_paths, output_dir):
    """
    Tạo biểu đồ nhiệt phân bố bounding boxes.

    Args:
        label_paths (List[str]): Danh sách đường dẫn đến các file nhãn.
        output_dir (str): Thư mục để lưu biểu đồ nhiệt.
    """
    # Kích thước heatmap nhỏ hơn để tối ưu hiệu suất
    heatmap_width, heatmap_height = 128, 72
    heatmap_data = np.zeros((heatmap_height, heatmap_width))

    # Tích lũy dữ liệu cho heatmap
    for label_path in tqdm(label_paths, desc="Generating bbox heatmap"):
        with open(label_path, "r") as file:
            for line in file:
                try:
                    # Đọc tọa độ bbox từ định dạng YOLO (class, x_center, y_center, width, height)
                    _, x_center_norm, y_center_norm, width_norm, height_norm = map(
                        float, line.strip().split()
                    )

                    # Chuyển đổi tọa độ normalized thành pixel trên heatmap
                    x_center = int(x_center_norm * heatmap_width)
                    y_center = int(y_center_norm * heatmap_height)
                    width = int(width_norm * heatmap_width)
                    height = int(height_norm * heatmap_height)

                    # Tính tọa độ bbox
                    x1 = max(0, x_center - width // 2)
                    x2 = min(heatmap_width, x_center + width // 2)
                    y1 = max(0, y_center - height // 2)
                    y2 = min(heatmap_height, y_center + height // 2)

                    # Tăng giá trị trong vùng bbox
                    heatmap_data[y1:y2, x1:x2] += 1
                except ValueError:
                    continue

    # Làm mịn heatmap bằng Gaussian filter
    heatmap_smoothed = gaussian_filter(heatmap_data, sigma=2)

    # Chuẩn hóa heatmap về khoảng [0,1]
    if np.max(heatmap_smoothed) > 0:
        heatmap_normalized = heatmap_smoothed / np.max(heatmap_smoothed)
    else:
        heatmap_normalized = heatmap_smoothed

    # Vẽ và lưu biểu đồ nhiệt
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_normalized, cmap="jet")
    plt.colorbar(label="Normalized Density")
    plt.title("Bounding Box Distribution Heatmap")
    plt.xlabel("Image Width")
    plt.ylabel("Image Height")
    plt.savefig(
        os.path.join(output_dir, "bbox_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"Đã lưu biểu đồ nhiệt phân bố bounding box vào {output_dir}/bbox_heatmap.png"
    )

    # Tính và in một số thống kê
    print("Thống kê phân bố bbox:")
    print(f"Tổng số bbox: {int(np.sum(heatmap_data))}")
    print(f"Độ tập trung trung bình: {np.mean(heatmap_normalized):.3f}")
    print(f"Độ tập trung cao nhất: {np.max(heatmap_normalized):.3f}")

    return heatmap_normalized


def analyze_image_colors(image_paths, output_dir):
    """Analyze image colors using parallel processing."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    thread_local = threading.local()

    def process_image(image_path):
        # Initialize thread-local OpenCV
        if not hasattr(thread_local, "cv2"):
            thread_local.cv2 = __import__("cv2")

        image = thread_local.cv2.imread(image_path)
        if image is None:
            return None

        hsv = thread_local.cv2.cvtColor(image, thread_local.cv2.COLOR_BGR2HSV)
        return {
            "brightness": np.mean(hsv[:, :, 2]),
            "saturation": np.mean(hsv[:, :, 1]),
            "contrast": np.std(hsv[:, :, 2]),
        }

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(
            tqdm(
                executor.map(process_image, image_paths),
                total=len(image_paths),
                desc="Analyzing image colors",
            )
        )

    # Filter out None results and separate values
    results = [r for r in results if r is not None]
    brightness_values = [r["brightness"] for r in results]
    saturation_values = [r["saturation"] for r in results]
    contrast_values = [r["contrast"] for r in results]

    # Plot distributions
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
        "Ban đêm": nighttime_images,
    }

    # Gọi hàm explore_images_combined
    explore_images_combined(image_paths_dict, output_dir)

    explore_labels(total_labels, "total", output_dir)
    explore_labels(daytime_labels, "daytime", output_dir)
    explore_labels(nighttime_labels, "nighttime", output_dir)

    save_random_images_with_boxes(
        daytime_images, daytime_folder, os.path.join(output_dir, "daytime_samples")
    )
    save_random_images_with_boxes(
        nighttime_images,
        nighttime_folder,
        os.path.join(output_dir, "nighttime_samples"),
    )

    analyze_box_dimensions(total_labels, output_dir)
    analyze_bbox_heatmap(total_labels, output_dir)
    analyze_image_colors(total_images, output_dir)


if __name__ == "__main__":
    main()
