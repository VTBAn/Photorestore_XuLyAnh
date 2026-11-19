import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def add_gaussian_noise(img: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Thêm nhiễu Gaussian trung bình 0, độ lệch chuẩn sigma (5..50)."""
    noise = np.random.normal(0.0, float(sigma), img.shape)
    out = img.astype(np.float64) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, percent: float = 5.0) -> np.ndarray:
    """Thêm nhiễu muối tiêu: p% điểm ảnh (một nửa = 255, một nửa = 0)."""
    out = img.copy()
    p = np.clip(float(percent) / 100.0, 0.0, 1.0)
    h, w = img.shape[:2]
    rng = np.random.rand(h, w)
    out[rng < p / 2.0] = 255
    out[(rng >= p / 2.0) & (rng < p)] = 0
    return out


def _geometric_mean_channel(ch: np.ndarray, ksize: int) -> np.ndarray:
    pad = ksize // 2
    padded = np.pad(ch.astype(np.float64) + 1.0, pad, mode="reflect")
    windows = sliding_window_view(padded, (ksize, ksize))
    # Tính bằng exp(mean(log(g + 1))) - 1 để tránh tràn số
    geo = np.exp(np.mean(np.log(windows), axis=(-2, -1))) - 1.0
    return np.clip(geo, 0, 255).astype(np.uint8)


def geometric_mean_filter_tc(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """[TC] Lọc trung bình hình học tự cài bằng NumPy."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    if img.ndim == 2:
        return _geometric_mean_channel(img, k)
    return np.stack([_geometric_mean_channel(img[:, :, c], k) for c in range(img.shape[2])], axis=2)


def _contraharmonic_channel(ch: np.ndarray, ksize: int, q: float) -> np.ndarray:
    pad = ksize // 2
    padded = np.pad(ch.astype(np.float64) + 1.0, pad, mode="reflect")
    windows = sliding_window_view(padded, (ksize, ksize))
    num = np.sum(np.power(windows, q + 1.0), axis=(-2, -1))
    den = np.sum(np.power(windows, q), axis=(-2, -1))
    res = (num / np.where(den == 0, 1e-12, den)) - 1.0
    return np.clip(res, 0, 255).astype(np.uint8)


def contraharmonic_filter_tc(img: np.ndarray, ksize: int = 3, q: float = 0.0) -> np.ndarray:
    """
    [TC] Lọc phản điều hòa bậc Q tự cài bằng NumPy.
    Công thức: sum(g^(Q+1)) / sum(g^Q). Q > 0 khử tiêu; Q < 0 khử muối.
    """
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    q = float(q)
    if img.ndim == 2:
        return _contraharmonic_channel(img, k, q)
    return np.stack([_contraharmonic_channel(img[:, :, c], k, q) for c in range(img.shape[2])], axis=2)


def compute_psnr(clean_img: np.ndarray, processed_img: np.ndarray) -> float:
    """Tính chỉ số PSNR (dB) giữa ảnh gốc sạch và ảnh sau xử lý."""
    if clean_img.shape != processed_img.shape:
        return 0.0
    mse = np.mean((clean_img.astype(np.float64) - processed_img.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))
