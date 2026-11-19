import cv2
import numpy as np


def split_hsv_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tách ảnh sang không gian HSV và trả về 3 kênh H, S, V dưới dạng ảnh xám."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.split(hsv)


def adjust_hsv(img: np.ndarray, h_shift: int = 0, s_scale: float = 1.0, v_scale: float = 1.0) -> np.ndarray:
    """
    Chỉnh Sắc độ (H), Bão hòa (S), Độ sáng (V):
    H_new = (H + h_shift) mod 180; S_new = clip(S * s_scale); V_new = clip(V * v_scale).
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(h_shift)) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(s_scale), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(v_scale), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def color_balance_bgr(img: np.ndarray, b_gain: float = 1.0, g_gain: float = 1.0, r_gain: float = 1.0) -> np.ndarray:
    """Cân bằng màu từng kênh B, G, R (hệ số 0.5..1.5). Khử ám vàng: tăng B, giảm R."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    out = bgr.astype(np.float32)
    out[:, :, 0] *= float(b_gain)
    out[:, :, 1] *= float(g_gain)
    out[:, :, 2] *= float(r_gain)
    return np.clip(out, 0, 255).astype(np.uint8)


COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "BONE": cv2.COLORMAP_BONE,
    "HOT": cv2.COLORMAP_HOT,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}


def apply_pseudocolor(img: np.ndarray, cmap_name: str = "JET") -> np.ndarray:
    """Tô màu giả cho ảnh xám bằng cv2.applyColorMap."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cmap = COLORMAPS.get(cmap_name, cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)
