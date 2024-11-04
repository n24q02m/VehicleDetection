import os
import numpy as np
from PIL import Image
from scipy.stats import beta

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_images_dir = os.path.join(base_path, 'images', 'train')
valid_images_dir = os.path.join(base_path, 'images', 'val')

# Hàm để tính toán thông số tối ưu cho mosaic
def compute_optimal_mosaic(image_dirs):
    mosaic_ratios = []
    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for _ in range(len(image_files) // 4):  # Chọn ngẫu nhiên bốn ảnh từ tập dữ liệu
            img1, img2, img3, img4 = np.random.choice(image_files, 4, replace=False)
            # Đọc các ảnh
            img1 = Image.open(os.path.join(image_dir, img1)).convert('RGB')
            img2 = Image.open(os.path.join(image_dir, img2)).convert('RGB')
            img3 = Image.open(os.path.join(image_dir, img3)).convert('RGB')
            img4 = Image.open(os.path.join(image_dir, img4)).convert('RGB')
            # Tính toán tỷ lệ ghép của các ảnh
            h, w = img1.size
            mosaic_ratio = (h * w) / (4 * h * w)  # Tỷ lệ ghép của các ảnh
            mosaic_ratios.append(mosaic_ratio)
    optimal_mosaic = np.mean(mosaic_ratios)
    return optimal_mosaic

# Hàm để tính toán thông số tối ưu cho mixup
def compute_optimal_mixup(image_dirs, alpha=0.2):
    mixup_ratios = []
    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for _ in range(len(image_files) // 2):  # Chọn ngẫu nhiên hai ảnh từ tập dữ liệu
            img1, img2 = np.random.choice(image_files, 2, replace=False)
            # Đọc các ảnh
            img1 = Image.open(os.path.join(image_dir, img1)).convert('RGB')
            img2 = Image.open(os.path.join(image_dir, img2)).convert('RGB')
            # Tạo trọng số ngẫu nhiên từ phân phối Beta
            lam = beta.rvs(alpha, alpha)
            mixup_ratios.append(lam)
    optimal_mixup = np.mean(mixup_ratios)
    return optimal_mixup

# Gộp cả hai tập dữ liệu train và val
all_image_dirs = [train_images_dir, valid_images_dir]

# Tính toán các thông số tối ưu
optimal_mosaic = compute_optimal_mosaic(all_image_dirs)
optimal_mixup = compute_optimal_mixup(all_image_dirs)

# In ra các thông số tối ưu
print("Thông số tối ưu cho mosaic:", optimal_mosaic)
print("Thông số tối ưu cho mixup:", optimal_mixup)

# Lưu kết quả các thông số tối ưu vào tệp TXT
output_stats_file = os.path.join('runs', 'augmentation-hyperparameter.txt')

# Đọc nội dung hiện tại của tệp
if os.path.exists(output_stats_file):
    with open(output_stats_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
else:
    lines = []

# Cập nhật hoặc thêm các thông số tối ưu
with open(output_stats_file, 'w', encoding='utf-8') as f:
    updated_mosaic = False
    updated_mixup = False
    for line in lines:
        if line.startswith("mosaic"):
            f.write(f"mosaic {optimal_mosaic}\n")
            updated_mosaic = True
        elif line.startswith("mixup"):
            f.write(f"mixup {optimal_mixup}\n")
            updated_mixup = True
        else:
            f.write(line)
    if not updated_mosaic:
        f.write(f"mosaic {optimal_mosaic}\n")
    if not updated_mixup:
        f.write(f"mixup {optimal_mixup}\n")

print(f"\nĐã lưu kết quả các thông số tối ưu vào tệp '{output_stats_file}'")