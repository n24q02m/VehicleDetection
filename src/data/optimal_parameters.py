import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

# Đường dẫn tới dataset
base_path = os.path.abspath(
    os.path.join(os.getcwd(), "data", "soict-hackathon-2024_dataset")
)
train_images_dir = os.path.join(base_path, "images", "train")
val_images_dir = os.path.join(base_path, "images", "val")
train_labels_dir = os.path.join(base_path, "labels", "train")
val_labels_dir = os.path.join(base_path, "labels", "val")

# Thư mục lưu kết quả
output_dir = os.path.join("runs", "explore")
os.makedirs(output_dir, exist_ok=True)

# Kích thước ảnh (giảm độ phân giải để tiết kiệm tài nguyên)
image_width, image_height = 256, 256

# Hàm để phân tích và hiển thị phân bố màu sắc
def analyze_color_distribution(image_dirs):
    # Sử dụng histogram với số bins cố định
    h_hist = np.zeros(256)
    s_hist = np.zeros(256)
    v_hist = np.zeros(256)

    for image_dir in image_dirs:
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert("HSV")
            image = image.resize((image_width, image_height))  # Giảm độ phân giải
            hsv_array = np.array(image)

            # Tính histogram cho từng kênh
            h_hist += np.histogram(hsv_array[:, :, 0], bins=256, range=(0, 255))[0]
            s_hist += np.histogram(hsv_array[:, :, 1], bins=256, range=(0, 255))[0]
            v_hist += np.histogram(hsv_array[:, :, 2], bins=256, range=(0, 255))[0]

    # Vẽ histogram cho từng kênh màu
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(256), h_hist, color="r")
    plt.title("Hue Distribution")
    plt.xlabel("Hue Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.bar(range(256), s_hist, color="g")
    plt.title("Saturation Distribution")
    plt.xlabel("Saturation Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.bar(range(256), v_hist, color="b")
    plt.title("Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "color_distribution.png"))
    plt.close()

    # Tính toán các thống kê
    h_values = np.repeat(np.arange(256), h_hist.astype(int))
    s_values = np.repeat(np.arange(256), s_hist.astype(int))
    v_values = np.repeat(np.arange(256), v_hist.astype(int))

    stats = {
        "h": {
            "mean": np.mean(h_values),
            "std": np.std(h_values),
            "min": np.min(h_values),
            "max": np.max(h_values),
            "25%": np.percentile(h_values, 25),
            "75%": np.percentile(h_values, 75),
            "iqr": np.percentile(h_values, 75) - np.percentile(h_values, 25),
        },
        "s": {
            "mean": np.mean(s_values),
            "std": np.std(s_values),
            "min": np.min(s_values),
            "max": np.max(s_values),
            "25%": np.percentile(s_values, 25),
            "75%": np.percentile(s_values, 75),
            "iqr": np.percentile(s_values, 75) - np.percentile(s_values, 25),
        },
        "v": {
            "mean": np.mean(v_values),
            "std": np.std(v_values),
            "min": np.min(v_values),
            "max": np.max(v_values),
            "25%": np.percentile(v_values, 25),
            "75%": np.percentile(v_values, 75),
            "iqr": np.percentile(v_values, 75) - np.percentile(v_values, 25),
        },
    }

    return stats

# Hàm để phân tích và hiển thị phân bố bounding boxes
def analyze_bounding_box_distribution(labels_dirs):
    # Giảm kích thước heatmap
    heatmap_width, heatmap_height = 128, 72
    heatmap_data = np.zeros((heatmap_height, heatmap_width))

    for labels_dir in labels_dirs:
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center_norm, y_center_norm, width_norm, height_norm = map(
                            float, parts[:5]
                        )
                        x_center = int(x_center_norm * heatmap_width)
                        y_center = int(y_center_norm * heatmap_height)
                        width = int(width_norm * heatmap_width)
                        height = int(height_norm * heatmap_height)
                        x1 = max(0, x_center - width // 2)
                        x2 = min(heatmap_width, x_center + width // 2)
                        y1 = max(0, y_center - height // 2)
                        y2 = min(heatmap_height, y_center + height // 2)
                        heatmap_data[y1:y2, x1:x2] += 1

    # Làm mịn heatmap
    heatmap_smoothed = gaussian_filter(heatmap_data, sigma=2)

    # Chuẩn hóa
    if np.max(heatmap_smoothed) > 0:
        heatmap_normalized = heatmap_smoothed / np.max(heatmap_smoothed)
    else:
        heatmap_normalized = heatmap_smoothed

    # Lưu heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_normalized, cmap="jet")
    plt.colorbar(label="Mật độ bounding boxes")
    plt.xlabel("Chiều rộng ảnh")
    plt.ylabel("Chiều cao ảnh")
    plt.title("Biểu đồ nhiệt phân bố bounding boxes")
    plt.savefig(os.path.join(output_dir, "bounding_box_heatmap.png"))
    plt.close()

    return heatmap_normalized

# Hàm tính toán tham số tối ưu cho hsv
def compute_optimal_hsv_parameters(stats):
    hsv_h = min(stats["h"]["iqr"] / 255.0, 0.5)
    hsv_s = min(stats["s"]["iqr"] / 255.0, 0.9)
    hsv_v = min(stats["v"]["iqr"] / 255.0, 0.9)
    return hsv_h, hsv_s, hsv_v

# Hàm tính toán tham số tối ưu cho erasing và crop_fraction
def compute_optimal_erasing_and_crop(heatmap):
    # Giả sử rằng các vùng có giá trị thấp trong heatmap là nơi có thể áp dụng erasing
    erasing = np.mean(1 - heatmap)
    # Giả sử rằng crop_fraction nên tập trung vào vùng có nhiều bounding boxes
    crop_fraction = np.clip(np.mean(heatmap) * 1.5, 0.1, 1.0)
    return erasing, crop_fraction

# Thực hiện phân tích
if __name__ == "__main__":
    # Phân tích phân bố màu sắc
    stats = analyze_color_distribution([train_images_dir, val_images_dir])

    # In thống kê màu sắc
    print("Thống kê màu sắc:")
    for channel in ["h", "s", "v"]:
        print(
            f"{channel.upper()} - Mean: {stats[channel]['mean']:.2f}, Std: {stats[channel]['std']:.2f}, Min: {stats[channel]['min']:.2f}, Max: {stats[channel]['max']:.2f}"
        )

    # Tính toán tham số hsv
    optimal_hsv_h, optimal_hsv_s, optimal_hsv_v = compute_optimal_hsv_parameters(stats)
    print("\nTham số HSV tối ưu:")
    print(f"hsv_h: {optimal_hsv_h:.3f}")
    print(f"hsv_s: {optimal_hsv_s:.3f}")
    print(f"hsv_v: {optimal_hsv_v:.3f}")

    # Phân tích phân bố bounding boxes
    heatmap = analyze_bounding_box_distribution([train_labels_dir, val_labels_dir])

    # Tính toán tham số erasing và crop_fraction
    optimal_erasing, optimal_crop_fraction = compute_optimal_erasing_and_crop(heatmap)
    print("\nTham số erasing và crop_fraction tối ưu:")
    print(f"erasing: {optimal_erasing:.3f}")
    print(f"crop_fraction: {optimal_crop_fraction:.3f}")

    # Lưu các tham số vào tệp
    output_stats_file = os.path.join("runs", "augmentation_parameters.txt")
    with open(output_stats_file, "w") as f:
        f.write(f"hsv_h {optimal_hsv_h:.3f}\n")
        f.write(f"hsv_s {optimal_hsv_s:.3f}\n")
        f.write(f"hsv_v {optimal_hsv_v:.3f}\n")
        f.write(f"erasing {optimal_erasing:.3f}\n")
        f.write(f"crop_fraction {optimal_crop_fraction:.3f}\n")

    print(f"\nĐã lưu các tham số tối ưu vào tệp '{output_stats_file}'.")
