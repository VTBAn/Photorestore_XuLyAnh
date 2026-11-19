import math
import cv2
import numpy as np


def sobel_edge(img: np.ndarray) -> np.ndarray:
    """Dò biên Sobel: độ lớn = sqrt(gx^2 + gy^2), chuẩn hóa về 0..255."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def log_edge(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Dò biên LoG: làm mờ Gaussian (sigma) rồi lấy Laplacian."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (0, 0), max(0.5, float(sigma)))
    lap = np.abs(cv2.Laplacian(blurred, cv2.CV_64F))
    return cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edge(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Dò biên Canny với 2 ngưỡng thấp và cao."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Canny(gray, int(low), int(high))


def sketch_effect(img: np.ndarray, low: int = 50, high: int = 150, soft_blur: bool = True) -> np.ndarray:
    """Hiệu ứng tranh phác thảo: 255 - Canny (nét đen trên nền trắng)."""
    edges = canny_edge(img, low, high)
    sketch = 255 - edges
    if soft_blur:
        sketch = cv2.GaussianBlur(sketch, (3, 3), 0.8)
    return sketch


def detect_lines_hough(img: np.ndarray, threshold: int = 80, min_length: int = 50,
                       max_gap: int = 10, canny_low: int = 50, canny_high: int = 150) -> tuple[np.ndarray, int, float | None]:
    """
    Tìm đường thẳng bằng cv2.HoughLinesP, vẽ đoạn thẳng màu đỏ lên ảnh.
    Trả về: (ảnh kết quả, số đoạn thẳng tìm được, góc đoạn dài nhất theo độ).
    """
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    edges = canny_edge(img, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(1, int(threshold)),
        minLineLength=max(1, int(min_length)),
        maxLineGap=max(0, int(max_gap))
    )

    if lines is None:
        return out, 0, None

    max_len = -1.0
    longest_angle = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        length = math.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            longest_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    return out, len(lines), longest_angle
