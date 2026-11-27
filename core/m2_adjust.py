import cv2
import numpy as np


def negative(img: np.ndarray) -> np.ndarray:
    """Ảnh âm bản: s = 255 - r."""
    return (255 - img.astype(np.float32)).clip(0, 255).astype(np.uint8)


def log_transform(img: np.ndarray) -> np.ndarray:
    """
    Biến đổi log: s = c * log(1 + r), với c = 255 / log(256).
    Làm sáng vùng tối, nén vùng sáng.
    """
    r = img.astype(np.float64)
    c = 255.0 / np.log(256.0)
    s = c * np.log1p(r)
    return np.clip(s, 0, 255).astype(np.uint8)


def gamma_transform(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Biến đổi gamma: s = 255 * (r / 255)^gamma.
    gamma < 1 làm sáng ảnh; gamma > 1 làm tối ảnh.
    """
    gamma = max(0.05, float(gamma))
    r = img.astype(np.float64) / 255.0
    s = 255.0 * np.power(r, gamma)
    return np.clip(s, 0, 255).astype(np.uint8)


def contrast_stretch(img: np.ndarray, low: int = 0, high: int = 255) -> np.ndarray:
    """
    Kéo giãn tương phản: s = (r - a) * 255 / (b - a), cắt về [0, 255].
    """
    a = float(min(low, high))
    b = float(max(low, high))
    if b - a < 1.0:
        b = a + 1.0
    s = (img.astype(np.float64) - a) * 255.0 / (b - a)
    return np.clip(s, 0, 255).astype(np.uint8)


def _equalize_single_channel_tc(channel: np.ndarray) -> np.ndarray:
    """Cân bằng histogram cho 1 kênh ảnh bằng NumPy."""
    # 1. Đếm số điểm ảnh ở mỗi mức xám 0..255
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
    # 2. Chia cho tổng số điểm ảnh -> xác suất
    pdf = hist / channel.size
    # 3. Cộng dồn -> hàm phân bố tích lũy CDF
    cdf = np.cumsum(pdf)
    # 4. Ánh xạ: s = round(255 * CDF(r))
    lut = np.round(255.0 * cdf).clip(0, 255).astype(np.uint8)
    return lut[channel]


def histogram_equalize_tc(img: np.ndarray) -> np.ndarray:
    """Cân bằng histogram tự cài bằng NumPy (hỗ trợ cả ảnh xám và ảnh màu)."""
    if img.ndim == 2:
        return _equalize_single_channel_tc(img)
    channels = cv2.split(img)
    eq_channels = [_equalize_single_channel_tc(ch) for ch in channels]
    return cv2.merge(eq_channels)


def histogram_equalize_opencv(img: np.ndarray) -> np.ndarray:
    """Cân bằng histogram dùng cv2.equalizeHist để so sánh với hàm [TC]."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    channels = cv2.split(img)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Cân bằng histogram thích nghi CLAHE (cv2.createCLAHE)."""
    tile_size = max(2, int(tile_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size))
    if img.ndim == 2:
        return clahe.apply(img)
    channels = cv2.split(img)
    out_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(out_channels)
