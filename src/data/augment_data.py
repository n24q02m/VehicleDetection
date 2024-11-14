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
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=15, p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.15, p=0.3
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 2.0), p=1.0),
                    A.GaussNoise(var_limit=(0.0, 1.56), p=1.0),
                ],
                p=0.3,
            ),
            A.RandomRain(p=0.3),
            A.RandomFog(p=0.3),
            A.RandomSnow(p=0.3),
            A.RandomShadow(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.RandomSunFlare(p=0.3),
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
    Apply data augmentation only to images with bounding boxes.
    Non-bbox images are preserved in their original location.
    """
    # Get list of images that have corresponding label files
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')}
    image_files = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 
        os.path.splitext(f)[0] in label_files
    ]

    augmentation_pipeline = get_augmentation_pipeline()
    
    # Process images in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    thread_local = threading.local()
    
    def process_image(args):
        image_file, aug_idx = args
        
        # Initialize thread-local OpenCV to avoid conflicts
        if not hasattr(thread_local, 'cv2'):
            thread_local.cv2 = __import__('cv2')
            
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')
        
        # Read image
        image = thread_local.cv2.imread(image_path)
        if image is None:
            return
            
        h, w = image.shape[:2]
        
        # Read labels
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    # Additional validation for bbox area and coordinates
                    if (x_max > x_min and y_max > y_min and 
                        width * height > 0 and
                        x_min >= 0 and y_min >= 0 and 
                        x_max <= w and y_max <= h):
                        bboxes.append([x_min, y_min, x_max, y_max])
                        class_labels.append(int(class_id))
        
        if not bboxes:
            return
            
        # Apply augmentation
        augmented = augmentation_pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        # Save augmented image and labels
        augmented_image_filename = f"{os.path.splitext(image_file)[0]}_aug_{aug_idx}.jpg"
        augmented_image_path = os.path.join(images_folder, augmented_image_filename)
        thread_local.cv2.imwrite(augmented_image_path, augmented['image'])
        
        # Convert bboxes back to YOLO format and save
        with open(os.path.join(labels_folder, f"{os.path.splitext(augmented_image_filename)[0]}.txt"), 'w') as f:
            for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                x_min, y_min, x_max, y_max = bbox
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
    
    # Create all augmentation tasks
    tasks = [(img, aug_idx) for img in image_files for aug_idx in range(augmentations_per_image)]
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(process_image, tasks),
            total=len(tasks),
            desc="Augmenting dataset"
        ))
    
    print(f"Đã tăng cường dữ liệu cho {len(image_files)} ảnh có bbox.")


def main():
    train_images_folder = "./data/soict-hackathon-2024_dataset/images/train"
    train_labels_folder = "./data/soict-hackathon-2024_dataset/labels/train"

    augment_dataset(train_images_folder, train_labels_folder)

if __name__ == "__main__":
    main()
