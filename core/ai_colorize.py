import os
import cv2
import numpy as np


class AIColorizer:
    """
    Phương án A: Tô màu tự động ảnh đen trắng bằng mô hình Zhang et al. (2016)
    sử dụng cv2.dnn.readNetFromCaffe.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.net = None

    def _get_paths(self) -> tuple[str, str, str]:
        proto = os.path.join(self.models_dir, "colorization_deploy_v2.prototxt")
        model = os.path.join(self.models_dir, "colorization_release_v2.caffemodel")
        pts = os.path.join(self.models_dir, "pts_in_hull.npy")
        return proto, model, pts

    def is_ready(self) -> tuple[bool, list[str]]:
        proto, model, pts = self._get_paths()
        missing = [os.path.basename(p) for p in (proto, model, pts) if not os.path.exists(p)]
        return len(missing) == 0, missing

    def load_model(self):
        if self.net is not None:
            return
        proto, model, pts_path = self._get_paths()
        net = cv2.dnn.readNetFromCaffe(proto, model)
        pts = np.load(pts_path)

        # Nạp các tâm cụm màu (ab cluster centers) vào lớp class8_ab và conv8_313_rh
        pts = pts.transpose().reshape(2, 313, 1, 1)
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        net.getLayer(class8).blobs = [pts.astype(np.float32)]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
        self.net = net

    def colorize(self, img: np.ndarray) -> np.ndarray:
        """Tô màu ảnh đen trắng (nhận ảnh xám hoặc BGR, trả về ảnh BGR đã tô màu)."""
        self.load_model()
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        scaled = bgr.astype(np.float32) / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize kênh L về 224x224 theo đầu vào mạng Caffe
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50.0

        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (bgr.shape[1], bgr.shape[0]))

        # Ghép kênh L gốc với kênh ab dự đoán rồi chuyển về BGR
        L_orig = cv2.split(lab)[0]
        colorized_lab = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr * 255.0, 0, 255).astype(np.uint8)
        return colorized_bgr
