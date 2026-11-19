import cv2
import numpy as np

SHAPES = {
    "Vuông (RECT)": cv2.MORPH_RECT,
    "Chữ thập (CROSS)": cv2.MORPH_CROSS,
    "Ellipse (ELLIPSE)": cv2.MORPH_ELLIPSE,
}


def binarize(img: np.ndarray, thresh: int = 127, use_otsu: bool = False, invert: bool = False) -> np.ndarray:
    """Nhị phân hóa ảnh bằng ngưỡng tay hoặc Otsu; tùy chọn đảo màu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, int(thresh), 255, mode)
    return binary


def morphology_op(img: np.ndarray, op: str = "open", shape_name: str = "Vuông (RECT)",
                  ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Thực hiện phép Co, Giãn, Mở, Đóng với phần tử cấu trúc tùy chọn."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    iters = max(1, int(iterations))
    shape = SHAPES.get(shape_name, cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (k, k))

    if op == "erode":
        return cv2.erode(img, kernel, iterations=iters)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=iters)
    if op == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iters)
    if op == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return img.copy()


def extract_boundary(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Trích biên hình thái: biên = ảnh - Co(ảnh)."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    eroded = cv2.erode(img, kernel, iterations=1)
    return cv2.subtract(img, eroded)


def remove_small_regions(img: np.ndarray, min_area: int = 50) -> tuple[np.ndarray, int]:
    """
    Xóa các vùng nhỏ (chấm bẩn) bằng cv2.connectedComponentsWithStats.
    Trả về: (ảnh đã làm sạch, số vùng đã xóa).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    removed_count = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            cleaned[labels == lbl] = 255
        else:
            removed_count += 1
    return cleaned, removed_count
