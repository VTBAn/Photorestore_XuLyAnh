import cv2
import numpy as np


def _ensure_odd(k: int) -> int:
    k = max(3, int(k))
    return k if k % 2 == 1 else k + 1


def mean_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Lọc trung bình (cv2.blur)."""
    k = _ensure_odd(ksize)
    return cv2.blur(img, (k, k))


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Lọc Gaussian (cv2.GaussianBlur)."""
    k = _ensure_odd(ksize)
    return cv2.GaussianBlur(img, (k, k), float(sigma))


def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Lọc trung vị (cv2.medianBlur)."""
    k = _ensure_odd(ksize)
    return cv2.medianBlur(img, k)


def sharpen_laplacian(img: np.ndarray, c: float = 0.5) -> np.ndarray:
    """Làm sắc nét bằng Laplacian: g = f - c * Laplacian(f)."""
    f = img.astype(np.float64)
    lap = cv2.Laplacian(f, cv2.CV_64F)
    g = f - float(c) * lap
    return np.clip(g, 0, 255).astype(np.uint8)


def unsharp_masking(img: np.ndarray, k: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Unsharp masking: mặt nạ = f - Gaussian(f); g = f + k * mặt nạ."""
    f = img.astype(np.float64)
    blurred = cv2.GaussianBlur(f, (0, 0), max(0.5, float(sigma)))
    mask = f - blurred
    g = f + float(k) * mask
    return np.clip(g, 0, 255).astype(np.uint8)
