import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_images_dir = os.path.join(base_path, 'images', 'train')
valid_images_dir = os.path.join(base_path, 'images', 'val')
train_labels_dir = os.path.join(base_path, 'labels', 'train')
valid_labels_dir = os.path.join(base_path, 'labels', 'val')

# Xác định kích thước ảnh (thay đổi theo kích thước ảnh của bạn)
image_width, image_height = 1280, 720  # Ví dụ kích thước ảnh là 1280x720

# Khởi tạo ma trận heatmap
heatmap_data = np.zeros((image_height, image_width))

# Hàm để cập nhật heatmap từ một thư mục nhãn
def update_heatmap_from_labels(labels_dir):
    global heatmap_data
    # Duyệt qua tất cả các file nhãn trong thư mục
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            with open(file_path, 'r') as f:
                annotations = f.readlines()
                for annotation in annotations:
                    # Đọc thông tin từ nhãn
                    class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, annotation.strip().split())
                    # Chuyển tọa độ từ YOLO format về tọa độ pixel
                    x_center = int(x_center_norm * image_width)
                    y_center = int(y_center_norm * image_height)
                    width = int(width_norm * image_width)
                    height = int(height_norm * image_height)
                    # Tính toán tọa độ góc của bounding box
                    x_min = x_center - width // 2
                    x_max = x_center + width // 2
                    y_min = y_center - height // 2
                    y_max = y_center + height // 2
                    # Đảm bảo tọa độ nằm trong kích thước ảnh
                    x_min = max(0, x_min)
                    x_max = min(image_width - 1, x_max)
                    y_min = max(0, y_min)
                    y_max = min(image_height - 1, y_max)
                    # Tăng giá trị trong vùng bounding box
                    heatmap_data[y_min:y_max+1, x_min:x_max+1] += 1

# Cập nhật heatmap từ tập train và val
update_heatmap_from_labels(train_labels_dir)
update_heatmap_from_labels(valid_labels_dir)

# Làm mịn heatmap_data bằng Gaussian filter
heatmap_smoothed = gaussian_filter(heatmap_data, sigma=15)

# Chuẩn hóa heatmap_data về khoảng [0, 1]
heatmap_normalized = heatmap_smoothed / np.max(heatmap_smoothed)

# Tính toán thông số tối ưu cho erasing
def compute_optimal_erasing(heatmap):
    # Tính tỷ lệ vùng có annotation thấp (giá trị heatmap nhỏ hơn ngưỡng)
    threshold = 0.2  # Ngưỡng để xác định vùng có annotation thấp
    low_annotation_area = np.sum(heatmap < threshold)
    total_area = heatmap.size
    erasing_ratio = low_annotation_area / total_area
    return erasing_ratio

# Tính toán thông số tối ưu cho crop_fraction
def compute_optimal_crop_fraction(heatmap):
    # Tính tỷ lệ vùng có annotation cao (giá trị heatmap lớn hơn ngưỡng)
    threshold = 0.5  # Ngưỡng để xác định vùng có annotation cao
    high_annotation_area = np.sum(heatmap > threshold)
    total_area = heatmap.size
    crop_fraction = high_annotation_area / total_area
    return crop_fraction

# Tính toán các thông số tối ưu
optimal_erasing = compute_optimal_erasing(heatmap_normalized)
optimal_crop_fraction = compute_optimal_crop_fraction(heatmap_normalized)

# In ra các thông số tối ưu
print("Thông số tối ưu cho erasing:", optimal_erasing)
print("Thông số tối ưu cho crop_fraction:", optimal_crop_fraction)

# Lưu kết quả các thông số tối ưu vào tệp TXT
output_stats_file = os.path.join('runs', 'augmentation-hyperparameter.txt')
with open(output_stats_file, 'a', encoding='utf-8') as f:
    f.write("\nThông số tối ưu cho erasing và crop_fraction:\n")
    f.write(f"erasing: {optimal_erasing:.4f}\n")
    f.write(f"crop-fraction: {optimal_crop_fraction:.4f}\n")

print(f"\nĐã lưu kết quả các thông số tối ưu vào tệp '{output_stats_file}'")