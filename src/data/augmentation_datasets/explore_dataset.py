import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_labels_dir = os.path.join(base_path, 'labels', 'train')
val_labels_dir = os.path.join(base_path, 'labels', 'val')

# Xác định kích thước ảnh
image_width, image_height = 1280, 720

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
                    x_center = x_center_norm * image_width
                    y_center = y_center_norm * image_height
                    width = width_norm * image_width
                    height = height_norm * image_height
                    # Tính toán tọa độ góc của bounding box
                    x_min = int(x_center - width / 2)
                    x_max = int(x_center + width / 2)
                    y_min = int(y_center - height / 2)
                    y_max = int(y_center + height / 2)
                    # Đảm bảo tọa độ nằm trong kích thước ảnh
                    x_min = max(0, x_min)
                    x_max = min(image_width - 1, x_max)
                    y_min = max(0, y_min)
                    y_max = min(image_height - 1, y_max)
                    # Tăng giá trị trong vùng bounding box
                    heatmap_data[y_min:y_max+1, x_min:x_max+1] += 1

# Cập nhật heatmap từ tập train và val
update_heatmap_from_labels(train_labels_dir)
update_heatmap_from_labels(val_labels_dir)

# Làm mịn heatmap_data bằng Gaussian filter
heatmap_smoothed = gaussian_filter(heatmap_data, sigma=15)

# Chuẩn hóa heatmap_data về khoảng [0, 1]
heatmap_normalized = heatmap_smoothed / np.max(heatmap_smoothed)

# Hiển thị heatmap để trực quan
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_normalized, cmap='jet')
plt.colorbar(label='Mật độ annotation')
plt.xlabel('Chiều rộng ảnh')
plt.ylabel('Chiều cao ảnh')
plt.title('Biểu đồ nhiệt mức phân bổ annotation trong khung hình')
plt.show()

# Chia ảnh thành lưới ô vuông (ví dụ: 8x8)
grid_size = 8
grid_height = image_height // grid_size
grid_width = image_width // grid_size

# Tính tổng giá trị heatmap trong mỗi ô
crop_candidates = []
for i in range(grid_size):
    for j in range(grid_size):
        x_start = j * grid_width
        y_start = i * grid_height
        x_end = x_start + grid_width
        y_end = y_start + grid_height
        cell_heat = heatmap_normalized[y_start:y_end, x_start:x_end].sum()
        crop_candidates.append({
            'x_start': x_start,
            'y_start': y_start,
            'x_end': x_end,
            'y_end': y_end,
            'heat': cell_heat
        })

# Sắp xếp các ô theo giá trị heat
crop_candidates.sort(key=lambda x: x['heat'], reverse=True)  # Sắp xếp giảm dần

# Lấy top N ô có giá trị cao nhất (hoặc thấp nhất tùy mục tiêu)
top_n = 5
selected_crops = crop_candidates[:top_n]

# Xuất thông số crop
print("Các thông số Crop đề xuất:")
for idx, crop in enumerate(selected_crops):
    print(f"Crop {idx + 1}: (x_start: {crop['x_start']}, y_start: {crop['y_start']}, width: {grid_width}, height: {grid_height})")