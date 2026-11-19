import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QFileDialog, QMessageBox, QComboBox, QProgressBar, QListWidget, QListWidgetItem
)

from core.m8_video import frame_difference, OpticalFlowTracker
from gui.image_canvas import numpy_to_qpixmap


class VideoStudioDialog(QDialog):
    """Cửa sổ xử lý Video gia đình (M8)."""

    frame_extracted = Signal(np.ndarray, str)  # Gửi khung hình về cửa sổ chính

    def __init__(self, video_path: str, history_ops: list[tuple[str, callable, dict]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhotoRestore — Video Studio (M8)")
        self.resize(1050, 700)

        self.video_path = video_path
        self.history_ops = history_ops  # Danh sách (tên_lệnh, hàm, kwargs) từ Lịch sử ảnh
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Không thể mở tệp video này!")

        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.current_idx = 0
        self.current_frame: np.ndarray | None = None
        self.prev_frame: np.ndarray | None = None
        self.tracker = OpticalFlowTracker(max_corners=100)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self._build_ui()
        self._read_frame_at(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Vùng hiển thị khung hình
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111111; border: 1px solid #3c3c3c;")
        self.video_label.setMinimumHeight(400)
        layout.addWidget(self.video_label, stretch=1)

        # Thanh tua & thông tin khung hình
        seek_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal, minimum=0, maximum=max(0, self.total_frames - 1))
        self.slider.sliderMoved.connect(self._on_seek)
        self.info_label = QLabel()
        seek_row.addWidget(self.slider, stretch=1)
        seek_row.addWidget(self.info_label)
        layout.addLayout(seek_row)

        # Điều khiển phát & chế độ phân tích chuyển động
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Phát")
        self.btn_play.clicked.connect(self._toggle_play)
        btn_prev = QPushButton("⏮ Khung trước", objectName="secondaryBtn", clicked=self._prev_frame)
        btn_next = QPushButton("Khung sau ⏭", objectName="secondaryBtn", clicked=self._next_frame)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Xem bình thường",
            "Phát hiện thay đổi (Frame Differencing)",
            "Theo vết chuyển động (Lucas-Kanade Optical Flow)",
        ])
        self.mode_combo.currentIndexChanged.connect(lambda _: self.tracker.reset())

        btn_extract = QPushButton("Trích khung sang Ảnh", objectName="accentBtn", clicked=self._extract_to_main)
        btn_save_frame = QPushButton("Lưu khung ảnh...", objectName="secondaryBtn", clicked=self._save_frame)

        for w in (self.btn_play, btn_prev, btn_next, QLabel("Chế độ:"), self.mode_combo, btn_extract, btn_save_frame):
            ctrl_row.addWidget(w)
        layout.addLayout(ctrl_row)

        # Khu vực Áp lệnh từ Lịch sử cho toàn bộ video & Xuất video
        export_box = QHBoxLayout()
        self.ops_list = QListWidget()
        self.ops_list.setMaximumHeight(95)
        for idx, (name, _, _) in enumerate(self.history_ops):
            item = QListWidgetItem(f"{idx + 1}. {name}")
            item.setCheckState(Qt.Checked)
            self.ops_list.addItem(item)
        if not self.history_ops:
            self.ops_list.addItem("(Chưa có lệnh chỉnh ảnh nào trong Lịch sử)")

        right_export = QVBoxLayout()
        btn_export = QPushButton("Áp lệnh đã chọn & Xuất Video...", clicked=self._export_video)
        self.progress = QProgressBar(minimum=0, maximum=100, value=0)
        right_export.addWidget(btn_export)
        right_export.addWidget(self.progress)

        export_box.addWidget(QLabel("Chuỗi lệnh từ Lịch sử:"))
        export_box.addWidget(self.ops_list, stretch=1)
        export_box.addLayout(right_export)
        layout.addLayout(export_box)

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ Phát")
        else:
            self.timer.start(max(15, int(1000.0 / self.fps)))
            self.btn_play.setText("⏸ Dừng")

    def _on_seek(self, pos: int):
        self.tracker.reset()
        self.prev_frame = None
        self._read_frame_at(pos)

    def _prev_frame(self):
        if self.current_idx > 0:
            self.tracker.reset()
            self.prev_frame = None
            self._read_frame_at(self.current_idx - 1)

    def _next_frame(self):
        if self.current_idx + 1 >= self.total_frames:
            self.timer.stop()
            self.btn_play.setText("▶ Phát")
            return
        self._read_frame_at(self.current_idx + 1, sequential=True)

    def _read_frame_at(self, idx: int, sequential: bool = False):
        if not sequential:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        self.prev_frame = self.current_frame
        self.current_frame = frame
        self.current_idx = idx

        mode = self.mode_combo.currentIndex()
        display = frame
        if mode == 1 and self.prev_frame is not None:
            display = frame_difference(frame, self.prev_frame)
        elif mode == 2:
            display = self.tracker.process_frame(frame)

        pix = numpy_to_qpixmap(display).scaled(
            self.video_label.width() - 10, self.video_label.height() - 10,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.info_label.setText(f"Khung {idx + 1} / {self.total_frames}  |  {self.fps:.1f} fps")

    def _extract_to_main(self):
        if self.current_frame is not None:
            self.frame_extracted.emit(self.current_frame.copy(), f"Trích khung #{self.current_idx + 1}")
            QMessageBox.information(self, "Đã trích khung", "Khung hình đã được đưa sang vùng hiển thị ảnh chính!")

    def _save_frame(self):
        if self.current_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu khung hình", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            ext = ".png" if path.lower().endswith(".png") else ".jpg"
            ok, buf = cv2.imencode(ext, self.current_frame)
            if ok:
                buf.tofile(path)

    def _export_video(self):
        selected_ops = []
        for i in range(len(self.history_ops)):
            if self.ops_list.item(i).checkState() == Qt.Checked:
                selected_ops.append(self.history_ops[i])

        out_path, _ = QFileDialog.getSaveFileName(self, "Xuất video", "", "MP4 (*.mp4);;AVI (*.avi)")
        if not out_path:
            return

        self.timer.stop()
        self.btn_play.setText("▶ Phát")

        cap_in = cv2.VideoCapture(self.video_path)
        w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.lower().endswith(".mp4") else "XVID"))
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))

        idx = 0
        while True:
            ok, frame = cap_in.read()
            if not ok:
                break
            proc = frame
            for _, func, kwargs in selected_ops:
                res = func(proc, **kwargs)
                proc = res[0] if isinstance(res, tuple) else res
            if proc.ndim == 2:
                proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            if proc.shape[1] != w or proc.shape[0] != h:
                proc = cv2.resize(proc, (w, h))
            writer.write(proc)
            idx += 1
            self.progress.setValue(int(idx * 100 / self.total_frames))

        cap_in.release()
        writer.release()
        self.progress.setValue(100)
        QMessageBox.information(self, "Hoàn tất", f"Đã xuất video thành công ({idx} khung):\n{out_path}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
