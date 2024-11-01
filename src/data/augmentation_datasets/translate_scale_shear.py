import os
import numpy as np
from PIL import Image

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_images_dir = os.path.join(base_path, 'images', 'train')
valid_images_dir = os.path.join(base_path, 'images', 'val')

# Xác định kích thước ảnh (thay đổi theo kích thước ảnh của bạn)
image_width, image_height = 1280, 720  # Ví dụ kích thước ảnh là 1280x720

# Hàm để tính toán thông số tối ưu cho translate
def compute_optimal_translate(image_dirs):
    translate_x_ratios = []
    translate_y_ratios = []
    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            # Tính toán tỷ lệ dịch chuyển tối đa theo chiều rộng và chiều cao
            translate_x = np.random.uniform(0.0, 1.0) * width
            translate_y = np.random.uniform(0.0, 1.0) * height
            translate_x_ratios.append(translate_x / width)
            translate_y_ratios.append(translate_y / height)
    optimal_translate_x = np.mean(translate_x_ratios)
    optimal_translate_y = np.mean(translate_y_ratios)
    return optimal_translate_x, optimal_translate_y

# Hàm để tính toán thông số tối ưu cho scale
def compute_optimal_scale(image_dirs):
    scales = []
    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            scale = np.random.uniform(0.8, 1.2)
            scales.append(scale)
    optimal_scale = np.mean(scales)
    return optimal_scale

# Hàm để tính toán thông số tối ưu cho shear
def compute_optimal_shear(image_dirs):
    shears = []
    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            shear = np.random.uniform(-180, 180)
            shears.append(shear)
    optimal_shear = np.mean(shears)
    return optimal_shear

# Gộp cả hai tập dữ liệu train và val
all_image_dirs = [train_images_dir, valid_images_dir]

# Tính toán các thông số tối ưu
optimal_translate = compute_optimal_translate(all_image_dirs)
optimal_scale = compute_optimal_scale(all_image_dirs)
optimal_shear = compute_optimal_shear(all_image_dirs)

# In ra các thông số tối ưu
print("Thông số tối ưu cho translate:", optimal_translate)
print("Thông số tối ưu cho scale:", optimal_scale)
print("Thông số tối ưu cho shear:", optimal_shear)

# Lưu kết quả các thông số tối ưu vào tệp TXT
output_stats_file = os.path.join('runs', 'augmentation-hyperparameter.txt')
with open(output_stats_file, 'a', encoding='utf-8') as f:
    f.write("\nThông số tối ưu cho translate, scale, và shear:\n")
    f.write(f"translate: {optimal_translate}\n")
    f.write(f"scale: {optimal_scale}\n")
    f.write(f"shear: {optimal_shear}\n")

print(f"\nĐã lưu kết quả các thông số tối ưu vào tệp '{output_stats_file}'")