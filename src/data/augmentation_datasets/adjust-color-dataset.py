import os
from PIL import Image
import numpy as np

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_images_dir = os.path.join(base_path, 'images', 'train')
valid_images_dir = os.path.join(base_path, 'images', 'val')

# Hàm để tính toán các thống kê của hsv_h, hsv_s, hsv_v cho một hình ảnh
def calculate_hsv_statistics(image):
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]

    h_channel = hsv_array[:, :, 0]
    s_channel = hsv_array[:, :, 1]
    v_channel = hsv_array[:, :, 2]

    # Tính các thống kê
    h_mean = h_channel.mean()
    h_std = h_channel.std()
    h_min = h_channel.min()
    h_max = h_channel.max()

    s_mean = s_channel.mean()
    s_std = s_channel.std()
    s_min = s_channel.min()
    s_max = s_channel.max()

    v_mean = v_channel.mean()
    v_std = v_channel.std()
    v_min = v_channel.min()
    v_max = v_channel.max()

    return (h_mean, h_std, h_min, h_max), (s_mean, s_std, s_min, s_max), (v_mean, v_std, v_min, v_max)

# Tính toán thống kê cho tập hình ảnh
def compute_hsv_statistics_for_dataset(image_dirs):
    h_values = []
    s_values = []
    v_values = []

    for image_dir in image_dirs:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')

            (h_mean, h_std, h_min, h_max), (s_mean, s_std, s_min, s_max), (v_mean, v_std, v_min, v_max) = calculate_hsv_statistics(image)

            h_values.append((h_mean, h_std, h_min, h_max))
            s_values.append((s_mean, s_std, s_min, s_max))
            v_values.append((v_mean, v_std, v_min, v_max))

    # Chuyển đổi danh sách thành mảng numpy để dễ tính toán
    h_stats = np.array(h_values)
    s_stats = np.array(s_values)
    v_stats = np.array(v_values)

    # Tính thống kê chung cho toàn bộ tập dữ liệu
    h_overall_mean = h_stats[:, 0].mean()
    h_overall_std = h_stats[:, 0].std()
    h_overall_min = h_stats[:, 2].min()
    h_overall_max = h_stats[:, 3].max()

    s_overall_mean = s_stats[:, 0].mean()
    s_overall_std = s_stats[:, 0].std()
    s_overall_min = s_stats[:, 2].min()
    s_overall_max = s_stats[:, 3].max()

    v_overall_mean = v_stats[:, 0].mean()
    v_overall_std = v_stats[:, 0].std()
    v_overall_min = v_stats[:, 2].min()
    v_overall_max = v_stats[:, 3].max()

    return {
        'h': {'mean': h_overall_mean, 'std': h_overall_std, 'min': h_overall_min, 'max': h_overall_max},
        's': {'mean': s_overall_mean, 'std': s_overall_std, 'min': s_overall_min, 'max': s_overall_max},
        'v': {'mean': v_overall_mean, 'std': v_overall_std, 'min': v_overall_min, 'max': v_overall_max},
    }

# Hàm để tính toán các thông số tối ưu cho hsv_h, hsv_s, hsv_v
def compute_optimal_hsv_parameters(stats):

    # Tính toán phạm vi cho hue (hsv_h)
    hsv_h = min(stats['h']['std'] * 2, 0.5)

    # Tính toán phạm vi cho saturation (hsv_s)
    hsv_s = min(stats['s']['std'] * 2, 0.9)

    # Tính toán phạm vi cho brightness/value (hsv_v)
    hsv_v = min(stats['v']['std'] * 2, 0.9)

    return hsv_h, hsv_s, hsv_v

# Tính toán thống kê cho tập train và val
stats = compute_hsv_statistics_for_dataset([train_images_dir, valid_images_dir])

# In ra các thống kê
print("Thống kê HSV cho toàn bộ tập dữ liệu:")
print(f"Hue - Mean: {stats['h']['mean']:.4f}, Std: {stats['h']['std']:.4f}, Min: {stats['h']['min']:.4f}, Max: {stats['h']['max']:.4f}")
print(f"Saturation - Mean: {stats['s']['mean']:.4f}, Std: {stats['s']['std']:.4f}, Min: {stats['s']['min']:.4f}, Max: {stats['s']['max']:.4f}")
print(f"Value - Mean: {stats['v']['mean']:.4f}, Std: {stats['v']['std']:.4f}, Min: {stats['v']['min']:.4f}, Max: {stats['v']['max']:.4f}")

# Tính toán các thông số tối ưu cho hsv_h, hsv_s, hsv_v
optimal_hsv_h, optimal_hsv_s, optimal_hsv_v = compute_optimal_hsv_parameters(stats)

# In ra các thông số tối ưu
print("\nCác thông số tối ưu cho tăng cường dữ liệu:")
print(f"hsv_h (Hue adjustment factor): {optimal_hsv_h:.4f}")
print(f"hsv_s (Saturation adjustment factor): {optimal_hsv_s:.4f}")
print(f"hsv_v (Brightness adjustment factor): {optimal_hsv_v:.4f}")

# Lưu kết quả các thông số thống kê vào tệp TXT
output_stats_file = os.path.join(base_path, 'hsv_statistics.txt')
with open(output_stats_file, 'w') as f:
    f.write("Thống kê HSV cho toàn bộ tập dữ liệu:\n")
    f.write(f"Hue - Mean: {stats['h']['mean']:.4f}, Std: {stats['h']['std']:.4f}, Min: {stats['h']['min']:.4f}, Max: {stats['h']['max']:.4f}\n")
    f.write(f"Saturation - Mean: {stats['s']['mean']:.4f}, Std: {stats['s']['std']:.4f}, Min: {stats['s']['min']:.4f}, Max: {stats['s']['max']:.4f}\n")
    f.write(f"Value - Mean: {stats['v']['mean']:.4f}, Std: {stats['v']['std']:.4f}, Min: {stats['v']['min']:.4f}, Max: {stats['v']['max']:.4f}\n")
    f.write("\nCác thông số tối ưu cho tăng cường dữ liệu:\n")
    f.write(f"hsv_h (Hue adjustment factor): {optimal_hsv_h:.4f}\n")
    f.write(f"hsv_s (Saturation adjustment factor): {optimal_hsv_s:.4f}\n")
    f.write(f"hsv_v (Brightness adjustment factor): {optimal_hsv_v:.4f}\n")

print(f"\nĐã lưu kết quả các thông số thống kê vào tệp '{output_stats_file}'")