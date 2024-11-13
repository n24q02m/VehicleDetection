import cv2
import numpy as np

def measure_brightness(image):
    """
    Đo lường độ sáng của ảnh bằng cách tính độ sáng trung bình.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return brightness

def measure_contrast(image):
    """
    Đo lường độ tương phản của ảnh bằng cách tính độ lệch chuẩn của độ sáng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast

def adaptive_brightness_contrast(image, target_brightness=130, target_contrast=50):
    """
    Điều chỉnh độ sáng và độ tương phản dựa trên điều kiện ánh sáng.
    target_brightness: Độ sáng mong muốn của ảnh.
    target_contrast: Độ tương phản mong muốn của ảnh.
    """
    current_brightness = measure_brightness(image)
    current_contrast = measure_contrast(image)

    # Điều chỉnh độ sáng (alpha) và độ tương phản (beta) dựa trên sự khác biệt
    brightness_adjustment = target_brightness / max(current_brightness, 1)
    contrast_adjustment = target_contrast / max(current_contrast, 1)

    # Áp dụng điều chỉnh
    adjusted = cv2.convertScaleAbs(image, alpha=brightness_adjustment, beta=contrast_adjustment)
    return adjusted

def adaptive_gamma_correction(image, target_brightness=130):
    """
    Điều chỉnh gamma để kiểm soát độ sáng của ảnh.
    target_brightness: Độ sáng mong muốn của ảnh.
    """
    current_brightness = measure_brightness(image)
    gamma = np.log(target_brightness / max(current_brightness, 1)) / np.log(2) if current_brightness > 0 else 1.0
    gamma = max(0.5, min(gamma, 2.5))  # Giới hạn gamma trong khoảng hợp lý

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Tăng cường độ tương phản bằng CLAHE nếu ảnh có độ sáng thấp.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def reduce_noise(image, h=15, hForColorComponents=15, templateWindowSize=7, searchWindowSize=21):
    """
    Giảm nhiễu trong ảnh.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, hForColorComponents, templateWindowSize, searchWindowSize)

def preprocess_image(image):
    """
    Thực hiện Real-time Adaptation trên ảnh đầu vào.
    """
    # Điều chỉnh độ sáng và độ tương phản
    image = adaptive_brightness_contrast(image)

    # Điều chỉnh gamma
    image = adaptive_gamma_correction(image)

    # Tăng cường độ tương phản với CLAHE
    image = adaptive_clahe(image)

    # Giảm nhiễu
    image = reduce_noise(image)

    return image