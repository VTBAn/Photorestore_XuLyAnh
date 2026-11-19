import cv2
import numpy as np


def get_image_info(img: np.ndarray) -> dict:
    """Trả về thông tin ảnh: rộng, cao, số kênh, kiểu dữ liệu, dung lượng chưa nén."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(img.dtype),
        "size_bytes": w * h * channels,
    }


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh màu BGR sang ảnh xám (M1)."""
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_image(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Cắt ảnh theo hình chữ nhật (x1, y1) - (x2, y2)."""
    h, w = img.shape[:2]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def resize_image(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Đổi kích thước ảnh bằng cv2.resize."""
    new_w = max(1, int(new_w))
    new_h = max(1, int(new_h))
    interp = cv2.INTER_AREA if (new_w < img.shape[1] and new_h < img.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def quantize_gray_levels(img: np.ndarray, levels: int = 16) -> np.ndarray:
    """
    [TC] Giảm mức xám bằng NumPy (M1).
    Công thức: bước = 256 / L; giá trị mới = (r // bước) * bước
    """
    levels = max(2, min(256, int(levels)))
    step = 256.0 / levels
    out = np.floor(img.astype(np.float32) / step) * step
    return np.clip(out, 0, 255).astype(np.uint8)
