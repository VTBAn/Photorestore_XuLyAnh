import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from gui.image_canvas import numpy_to_qpixmap


class MultiImageDialog(QDialog):
    """Hộp thoại hiển thị 2 hoặc 3 ảnh cạnh nhau (So sánh Trước/Sau, So sánh OpenCV, Kênh HSV)."""

    def __init__(self, title: str, items: list[tuple[str, np.ndarray]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(min(1200, 380 * len(items)), 480)

        main_layout = QVBoxLayout(self)
        row = QHBoxLayout()

        for caption, img in items:
            col = QVBoxLayout()
            lbl_title = QLabel(caption)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; color: #38bdf8; padding: 4px;")

            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            pix = numpy_to_qpixmap(img).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_img.setPixmap(pix)

            col.addWidget(lbl_title)
            col.addWidget(lbl_img, stretch=1)
            row.addLayout(col)

        main_layout.addLayout(row)
        btn_close = QPushButton("Đóng", clicked=self.accept)
        btn_close.setFixedWidth(120)
        main_layout.addWidget(btn_close, alignment=Qt.AlignCenter)
