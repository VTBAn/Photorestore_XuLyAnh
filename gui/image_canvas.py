import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel, QScrollArea


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Chuyển mảng NumPy (BGR hoặc Grayscale uint8) sang QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class ImageCanvas(QScrollArea):
    """Vùng hiển thị ảnh hỗ trợ Zoom, cuộn, tọa độ & giá trị BGR, Cắt vùng và Ống hút màu."""

    pixel_hovered = Signal(int, int, str)
    pixel_picked = Signal(int, int, str)
    roi_selected = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #141414; border: none;")

        self.image_label = QLabel("Mở ảnh (Ctrl+O) để bắt đầu")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #71717a; font-size: 15px;")
        self.image_label.setMouseTracking(True)
        self.setWidget(self.image_label)

        self._img: np.ndarray | None = None
        self._zoom: float = 1.0
        self._tool: str = "pointer"
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None

        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseReleaseEvent = self._on_mouse_release

    def set_tool(self, tool_name: str):
        self._tool = tool_name
        cursor = Qt.CrossCursor if tool_name in ("crop", "eyedropper") else Qt.ArrowCursor
        self.image_label.setCursor(cursor)

    def set_image(self, img: np.ndarray | None):
        self._img = img
        self._render()

    def set_zoom(self, factor: float):
        self._zoom = max(0.1, min(6.0, float(factor)))
        self._render()

    def get_zoom(self) -> float:
        return self._zoom

    def wheelEvent(self, event):
        if self._img is not None and (event.modifiers() & Qt.ControlModifier):
            delta = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.set_zoom(self._zoom * delta)
        else:
            super().wheelEvent(event)

    def _render(self):
        if self._img is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Mở ảnh (Ctrl+O) để bắt đầu")
            return

        pixmap = numpy_to_qpixmap(self._img)
        w = max(1, int(pixmap.width() * self._zoom))
        h = max(1, int(pixmap.height() * self._zoom))
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self._tool == "crop" and self._drag_start and self._drag_end:
            painter = QPainter(scaled)
            painter.setPen(QPen(QColor("#38bdf8"), 2, Qt.DashLine))
            painter.drawRect(QRect(self._drag_start, self._drag_end))
            painter.end()

        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _to_image_coords(self, pos: QPoint) -> tuple[int, int] | None:
        if self._img is None:
            return None
        x = int(pos.x() / self._zoom)
        y = int(pos.y() / self._zoom)
        h, w = self._img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def _format_pixel(self, x: int, y: int) -> str:
        val = self._img[y, x]
        if self._img.ndim == 2:
            return f"Gray = {int(val)}"
        return f"(B, G, R) = ({int(val[0])}, {int(val[1])}, {int(val[2])})"

    def _on_mouse_move(self, event):
        coords = self._to_image_coords(event.position().toPoint())
        if coords:
            x, y = coords
            self.pixel_hovered.emit(x, y, self._format_pixel(x, y))
        if self._tool == "crop" and self._drag_start:
            self._drag_end = event.position().toPoint()
            self._render()

    def _on_mouse_press(self, event):
        if event.button() != Qt.LeftButton or self._img is None:
            return
        pt = event.position().toPoint()
        if self._tool == "crop":
            self._drag_start = pt
            self._drag_end = pt
        elif self._tool == "eyedropper":
            coords = self._to_image_coords(pt)
            if coords:
                x, y = coords
                self.pixel_picked.emit(x, y, self._format_pixel(x, y))

    def _on_mouse_release(self, event):
        if self._tool == "crop" and self._drag_start and self._drag_end:
            c1 = self._to_image_coords(self._drag_start)
            c2 = self._to_image_coords(self._drag_end)
            self._drag_start = None
            self._drag_end = None
            self._render()
            if c1 and c2 and (abs(c2[0] - c1[0]) > 4 and abs(c2[1] - c1[1]) > 4):
                self.roi_selected.emit(c1[0], c1[1], c2[0], c2[1])
