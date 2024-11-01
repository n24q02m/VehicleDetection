import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ExifTags
import torchvision.transforms as transforms

# Đường dẫn tới dataset
base_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'soict-hackathon-2024_dataset'))
train_images_dir = os.path.join(base_path, 'images', 'train')
valid_images_dir = os.path.join(base_path, 'images', 'val')
train_labels_dir = os.path.join(base_path, 'labels', 'train')
valid_labels_dir = os.path.join(base_path, 'labels', 'val')

# Kích thước đầu vào của mô hình YOLOv8s
input_size = (640, 640)

# Hàm để sửa hướng ảnh dựa trên dữ liệu EXIF
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            # Tìm mã cho tag 'Orientation'
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif_orientation = exif.get(orientation, None)
            if exif_orientation == 3:
                image = image.rotate(180, expand=True)
                angle = 180
            elif exif_orientation == 6:
                image = image.rotate(-90, expand=True)
                angle = -90
            elif exif_orientation == 8:
                image = image.rotate(90, expand=True)
                angle = 90
            else:
                angle = 0
        else:
            angle = 0
    except (AttributeError, KeyError, IndexError):
        angle = 0
    return image, angle

# Hàm để xoay bounding box
def rotate_bounding_boxes(boxes, angle, image_size):
    angle_rad = np.deg2rad(angle)
    image_width, image_height = image_size
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width_box, height_box = box
        # Chuyển từ tọa độ trung tâm sang x_min, y_min, x_max, y_max
        x_min = x_center - width_box / 2
        y_min = y_center - height_box / 2
        x_max = x_center + width_box / 2
        y_max = y_center + height_box / 2

        # Lấy các điểm góc của bounding box
        points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

        # Tính ma trận xoay
        cx, cy = image_width / 2, image_height / 2
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # Xoay các điểm
        points_centered = points - np.array([cx, cy])
        points_rotated = np.dot(points_centered, R.T) + np.array([cx, cy])

        # Lấy bounding box mới
        x_rotated = points_rotated[:, 0]
        y_rotated = points_rotated[:, 1]
        x_min_new = x_rotated.min()
        y_min_new = y_rotated.min()
        x_max_new = x_rotated.max()
        y_max_new = y_rotated.max()

        # Chuyển lại về tọa độ trung tâm
        x_center_new = (x_min_new + x_max_new) / 2
        y_center_new = (y_min_new + y_max_new) / 2
        width_box_new = x_max_new - x_min_new
        height_box_new = y_max_new - y_min_new

        new_boxes.append((class_id, x_center_new, y_center_new, width_box_new, height_box_new))
    return new_boxes

# Hàm để thay đổi kích thước hình ảnh và cập nhật bounding box
def resize_image_and_update_labels(image_path, label_path, output_size):
    # Đọc ảnh và lấy kích thước ban đầu
    image = Image.open(image_path)
    original_width, original_height = image.size

    # Sửa hướng ảnh dựa trên EXIF
    image, angle = correct_image_orientation(image)

    # Đọc và cập nhật tọa độ bounding box
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.strip().split())
            x_center = x_center_norm * original_width
            y_center = y_center_norm * original_height
            width_box = width_norm * original_width
            height_box = height_norm * original_height
            boxes.append((int(class_id), x_center, y_center, width_box, height_box))

        # Xoay bounding box nếu cần
        if angle != 0:
            boxes = rotate_bounding_boxes(boxes, angle, (original_width, original_height))

    # Thay đổi kích thước ảnh
    image = image.resize(output_size)
    new_width, new_height = image.size
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Cập nhật bounding box theo kích thước mới
    boxes = [(class_id,
              x_center * scale_x,
              y_center * scale_y,
              width_box * scale_x,
              height_box * scale_y)
             for class_id, x_center, y_center, width_box, height_box in boxes]

    return image, boxes

# Hàm hiển thị ảnh với bounding box
def display_images_with_boxes(images, boxes_list, titles=None):
    plt.figure(figsize=(15, 5))
    for i, (image, boxes) in enumerate(zip(images, boxes_list)):
        draw = ImageDraw.Draw(image)
        for box in boxes:
            class_id, x_center, y_center, width_box, height_box = box
            x_min = x_center - width_box / 2
            y_min = y_center - height_box / 2
            x_max = x_center + width_box / 2
            y_max = y_center + height_box / 2
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min), str(class_id), fill="yellow")
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Hàm xử lý và hiển thị ảnh với bounding box
def process_and_display_images_with_boxes(image_dir, label_dir, num_images=2):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random_images = random.sample(image_files, min(num_images, len(image_files)))
    images = []
    boxes_list = []
    for image_file in random_images:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        image, boxes = resize_image_and_update_labels(image_path, label_path, input_size)
        images.append(image)
        boxes_list.append(boxes)
    display_images_with_boxes(images, boxes_list, titles=random_images)

# Hiển thị 1-2 ảnh từ tập train với bounding box sau khi thay đổi kích thước
print('Hiển thị 1-2 ảnh từ tập train với bounding box:')
process_and_display_images_with_boxes(train_images_dir, train_labels_dir)

# Hiển thị 1-2 ảnh từ tập val với bounding box sau khi thay đổi kích thước
print('Hiển thị 1-2 ảnh từ tập val với bounding box:')
process_and_display_images_with_boxes(valid_images_dir, valid_labels_dir)