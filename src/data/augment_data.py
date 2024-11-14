import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A


def get_augmentation_pipeline():
    """
    Định nghĩa quy trình tăng cường dữ liệu sử dụng Albumentations.
    """
    return A.Compose(
        [
            A.Mosaic(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                scale_limit=0.3, rotate_limit=15, shear_limit=15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.15, p=0.5
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.GaussNoise(var_limit=(10, 40), p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=10,
                max_height=0.1,
                max_width=0.1,
                p=0.5,
            ),
            A.RandomRain(p=0.3),
            A.RandomFog(p=0.3),
            A.RandomSnow(p=0.2),
            A.RandomShadow(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomSunFlare(p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.4,
        ),
    )


def augment_dataset(images_folder, labels_folder, augmentations_per_image=2):
    """
    Áp dụng tăng cường dữ liệu cho bộ dữ liệu.

    Args:
        images_folder (str): Thư mục chứa ảnh gốc.
        labels_folder (str): Thư mục chứa nhãn tương ứng.
        augmentations_per_image (int): Số lượng phiên bản tăng cường cho mỗi ảnh.
    """
    image_files = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    num_images = len(image_files)

    for idx in tqdm(range(num_images), desc="Augmenting dataset"):
        image_file = image_files[idx]
        image_path = os.path.join(images_folder, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_file)

        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            bboxes = []
            class_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    bboxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(int(class_id))

            for i in range(augmentations_per_image):
                # Chuẩn bị dữ liệu cho Mosaic transformation
                if any(isinstance(t, A.Mosaic) for t in get_augmentation_pipeline().transforms):
                    # Chọn thêm ba ảnh và nhãn ngẫu nhiên
                    other_indices = [i for i in range(num_images) if i != idx]
                    mosaic_indices = np.random.choice(other_indices, 3, replace=False)
                    images = [image]
                    bboxes_list = [bboxes]
                    class_labels_list = [class_labels]

                    for mosaic_idx in mosaic_indices:
                        mosaic_image_file = image_files[mosaic_idx]
                        mosaic_image_path = os.path.join(images_folder, mosaic_image_file)
                        mosaic_label_file = os.path.splitext(mosaic_image_file)[0] + '.txt'
                        mosaic_label_path = os.path.join(labels_folder, mosaic_label_file)

                        mosaic_image = cv2.imread(mosaic_image_path)
                        mosaic_h, mosaic_w = mosaic_image.shape[:2]

                        mosaic_bboxes = []
                        mosaic_class_labels = []
                        if os.path.exists(mosaic_label_path):
                            with open(mosaic_label_path, 'r') as f:
                                for line in f:
                                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                                    x_center *= mosaic_w
                                    y_center *= mosaic_h
                                    width *= mosaic_w
                                    height *= mosaic_h
                                    x_min = x_center - width / 2
                                    y_min = y_center - height / 2
                                    x_max = x_center + width / 2
                                    y_max = y_center + height / 2
                                    mosaic_bboxes.append([x_min, y_min, x_max, y_max])
                                    mosaic_class_labels.append(int(class_id))

                        images.append(mosaic_image)
                        bboxes_list.append(mosaic_bboxes)
                        class_labels_list.append(mosaic_class_labels)

                    # Tạo dictionary cho additional targets
                    additional_images = {
                        f"image{i}": images[i] for i in range(1, 4)
                    }
                    additional_bboxes = {
                        f"bboxes{i}": bboxes_list[i] for i in range(1, 4)
                    }
                    additional_class_labels = {
                        f"class_labels{i}": class_labels_list[i] for i in range(1, 4)
                    }

                    # Cập nhật hàm Compose để nhận additional targets
                    transforms = get_augmentation_pipeline()
                    for i in range(1, 4):
                        transforms.add_targets({f"image{i}": "image"})
                        transforms.add_targets({f"bboxes{i}": "bboxes"})
                        transforms.add_targets({f"class_labels{i}": "class_labels"})

                    # Áp dụng phép biến đổi
                    transformed = transforms(
                        image=images[0],
                        bboxes=bboxes_list[0],
                        class_labels=class_labels_list[0],
                        **additional_images,
                        **additional_bboxes,
                        **additional_class_labels,
                    )

                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_class_labels = transformed['class_labels']
                else:
                    transformed = get_augmentation_pipeline()(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_class_labels = transformed['class_labels']

                # Chuyển bounding boxes về định dạng YOLO
                aug_bboxes_yolo = []
                for bbox in aug_bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / w
                    y_center = (y_min + y_max) / 2 / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    aug_bboxes_yolo.append([x_center, y_center, width, height])

                # Lưu ảnh và nhãn tăng cường
                aug_image_name = f"aug_{i}_{image_file}"
                aug_label_name = f"aug_{i}_{label_file}"
                cv2.imwrite(os.path.join(images_folder, aug_image_name), aug_image)
                with open(os.path.join(labels_folder, aug_label_name), 'w') as f:
                    for bbox, class_id in zip(aug_bboxes_yolo, aug_class_labels):
                        bbox_str = ' '.join(map(str, bbox))
                        f.write(f"{class_id} {bbox_str}\n")

    print("Đã tăng cường dữ liệu.")


def main():
    train_images_folder = "./data/soict-hackathon-2024_dataset/images/train"
    train_labels_folder = "./data/soict-hackathon-2024_dataset/labels/train"

    augment_dataset(train_images_folder, train_labels_folder)

if __name__ == "__main__":
    main()