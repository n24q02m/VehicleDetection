import cv2
import numpy as np

def adjust_brightness_contrast(image, alpha=1.4, beta=10):
    """
    Điều chỉnh độ sáng và độ tương phản của ảnh.
    alpha: Hệ số tăng cường độ sáng (1.0-3.0)
    beta: Giá trị cộng thêm vào mỗi pixel (0-100)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def reduce_noise(image, h=15, hForColorComponents=15, templateWindowSize=7, searchWindowSize=21):
    """
    Giảm nhiễu trong ảnh.
    h: Tham số lọc. Giá trị càng cao thì mức độ giảm nhiễu càng cao.
    hForColorComponents: Tham số lọc cho các thành phần màu. Giá trị càng cao thì mức độ giảm nhiễu càng cao.
    templateWindowSize: Kích thước cửa sổ mẫu. Giá trị càng lớn thì mức độ giảm nhiễu càng cao.
    searchWindowSize: Kích thước cửa sổ tìm kiếm. Giá trị càng lớn thì mức độ giảm nhiễu càng cao.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, hForColorComponents, templateWindowSize, searchWindowSize)

def sharpen_image(image):
    """
    Làm sắc nét ảnh.
    """
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_clahe(image, clip_limit=2.5, tile_grid_size=(16, 16)):
    """
    Tăng cường độ tương phản của ảnh bằng CLAHE.
    clip_limit: Giá trị cắt ngưỡng của histogram.
    tile_grid_size: Kích thước của lưới mà CLAHE sẽ áp dụng.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def gamma_correction(image, gamma=1.0):
    """
    Điều chỉnh gamma của ảnh.
    gamma: Giá trị gamma (0.1-3.0)
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def mask_bright_spots(image, threshold=240):
    """
    Tạo mặt nạ để giảm độ sáng của các vùng quá chói.
    threshold: Ngưỡng để xác định các vùng sáng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    image_masked = cv2.bitwise_and(image, image, mask=mask_inv)
    return image_masked

def preprocess_image(image):
    # Tạo mặt nạ để giảm độ sáng của các vùng quá chói
    image = mask_bright_spots(image)
    
    # Tăng cường độ tương phản bằng CLAHE
    image = apply_clahe(image)
    
    # Điều chỉnh gamma
    image = gamma_correction(image, gamma=1.5)
    
    # Giảm nhiễu
    image = reduce_noise(image)
    
    # Làm sắc nét
    image = sharpen_image(image)
    
    return image