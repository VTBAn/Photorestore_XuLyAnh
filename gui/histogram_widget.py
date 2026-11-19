import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramWidget(QWidget):
    """Bảng Histogram tự cập nhật (1 đường cho ảnh xám, 3 đường B/G/R cho ảnh màu)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(3.8, 2.0), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        self._style_axes()

    def _style_axes(self):
        self.ax.set_facecolor("#252526")
        self.ax.tick_params(colors="#a1a1aa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#3c3c3c")
        self.ax.set_xlim([0, 255])

    def update_histogram(self, img: np.ndarray | None):
        self.ax.clear()
        self._style_axes()

        if img is not None:
            if img.ndim == 2:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                self.ax.plot(hist, color="#e4e4e7", linewidth=1.2)
            else:
                for i, col in enumerate(("#3b82f6", "#22c55e", "#ef4444")):
                    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                    self.ax.plot(hist, color=col, linewidth=1.2)

        self.figure.tight_layout(pad=0.4)
        self.canvas.draw_idle()
