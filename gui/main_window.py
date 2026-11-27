import os
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QListWidget, QTextEdit, QToolBar, QDialog, QSpinBox, QCheckBox,
    QFormLayout, QComboBox
)

from core import m1_basic, m2_adjust, m3_filter, m4_noise, m5_color, m6_morph, m7_edge
from core.ai_colorize import AIColorizer
from gui.image_canvas import ImageCanvas
from gui.histogram_widget import HistogramWidget
from gui.dialogs import MultiImageDialog
from gui.video_dialog import VideoStudioDialog
from gui.styles import DARK_THEME_QSS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoRestore — Phần mềm Phục hồi & Chỉnh sửa Ảnh cũ")
        self.resize(1400, 860)
        self.setStyleSheet(DARK_THEME_QSS)

        # Trạng thái ảnh & Lịch sử Undo/Redo (G3)
        self.original_img: np.ndarray | None = None
        self.current_img: np.ndarray | None = None
        self.preview_img: np.ndarray | None = None
        self.clean_img_for_psnr: np.ndarray | None = None  # Lưu ảnh trước khi thêm nhiễu (M4)
        self.current_filepath: str | None = None

        # Mỗi mục lịch sử: (tên_lệnh, ảnh_kết_quả, hàm_core_hoặc_None, kwargs)
        self.history_stack: list[tuple[str, np.ndarray, callable | None, dict]] = []
        self.history_index: int = -1
        self.max_history: int = 25

        # Bộ tô màu AI (Phương án A)
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.colorizer = AIColorizer(models_dir)

        # Vùng hiển thị ảnh chính (G1)
        self.canvas = ImageCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.pixel_hovered.connect(self._update_status_pixel)
        self.canvas.pixel_picked.connect(self._on_eyedropper_picked)
        self.canvas.roi_selected.connect(self._on_crop_selected)

        self._init_menus()
        self._init_toolbar()
        self._init_docks()
        self._init_statusbar()

    # =========================================================================
    # MENU BAR (G1, G4, M1–M8, AI)
    # =========================================================================
    def _init_menus(self):
        mb = self.menuBar()

        # 1. TỆP (G4)
        m_file = mb.addMenu("Tệp")
        self._act(m_file, "Mở ảnh...", self.open_image, "Ctrl+O")
        self._act(m_file, "Lưu", self.save_image, "Ctrl+S")
        self._act(m_file, "Lưu thành (PNG/JPG)...", self.save_image_as, "Ctrl+Shift+S")
        self._act(m_file, "Mở video...", self.open_video, "Ctrl+Shift+O")
        m_file.addSeparator()
        self._act(m_file, "Hoàn tác (Undo)", self.undo, QKeySequence.Undo)
        self._act(m_file, "Làm lại (Redo)", self.redo, QKeySequence.Redo)
        self._act(m_file, "Khôi phục ảnh gốc", self.reset_original)
        m_file.addSeparator()
        self._act(m_file, "Thoát", self.close, "Ctrl+Q")

        # 2. ẢNH (M1)
        m_img = mb.addMenu("Ảnh")
        self._act(m_img, "Thông tin ảnh", self.cmd_image_info)
        self._act(m_img, "Chuyển ảnh xám", self.cmd_grayscale)
        self._act(m_img, "Cắt theo vùng chọn (Chọn công cụ Cắt)", lambda: self._select_tool("crop"))
        self._act(m_img, "Đổi kích thước...", self.cmd_resize_dialog)
        self._act(m_img, "Giảm mức xám", self.cmd_quantize)

        # 3. ĐIỀU CHỈNH (M2)
        m_adj = mb.addMenu("Điều chỉnh")
        self._act(m_adj, "Âm bản", self.cmd_negative)
        self._act(m_adj, "Biến đổi Log", self.cmd_log)
        self._act(m_adj, "Biến đổi Gamma", self.cmd_gamma)
        self._act(m_adj, "Kéo giãn tương phản", self.cmd_contrast_stretch)
        self._act(m_adj, "Cân bằng Histogram", self.cmd_hist_eq)
        self._act(m_adj, "CLAHE (Cân bằng thích nghi)", self.cmd_clahe)

        # 4. BỘ LỌC (M3 & M4)
        m_flt = mb.addMenu("Bộ lọc")
        self._act(m_flt, "Làm mịn Trung bình", self.cmd_mean_blur)
        self._act(m_flt, "Làm mịn Gaussian", self.cmd_gaussian_blur)
        self._act(m_flt, "Làm mịn Trung vị", self.cmd_median_blur)
        m_flt.addSeparator()
        self._act(m_flt, "Sắc nét Laplacian", self.cmd_laplacian_sharpen)
        self._act(m_flt, "Unsharp Masking", self.cmd_unsharp)
        m_flt.addSeparator()
        sub_noise = m_flt.addMenu("Nhiễu & Giảm nhiễu")
        self._act(sub_noise, "Thêm nhiễu Gaussian", self.cmd_add_gaussian_noise)
        self._act(sub_noise, "Thêm nhiễu muối tiêu", self.cmd_add_sp_noise)
        self._act(sub_noise, "Giảm nhiễu & Tính PSNR", self.cmd_denoise)

        # 5. MÀU (M5)
        m_col = mb.addMenu("Màu")
        self._act(m_col, "Xem kênh H, S, V", self.cmd_view_hsv)
        self._act(m_col, "Sắc độ / Bão hòa / Độ sáng (HSV)", self.cmd_adjust_hsv)
        self._act(m_col, "Cân bằng màu (B, G, R)", self.cmd_color_balance)
        self._act(m_col, "Tô màu giả (Colormap)", self.cmd_pseudocolor)

        # 6. HÌNH THÁI (M6)
        m_mor = mb.addMenu("Hình thái")
        self._act(m_mor, "Nhị phân hóa (Ngưỡng tay / Otsu)", self.cmd_binarize)
        self._act(m_mor, "Co (Erosion)", lambda: self.cmd_morph("erode", "Co (Erosion)"))
        self._act(m_mor, "Giãn (Dilation)", lambda: self.cmd_morph("dilate", "Giãn (Dilation)"))
        self._act(m_mor, "Mở (Opening)", lambda: self.cmd_morph("open", "Mở (Opening)"))
        self._act(m_mor, "Đóng (Closing)", lambda: self.cmd_morph("close", "Đóng (Closing)"))
        self._act(m_mor, "Trích biên hình thái", self.cmd_boundary)
        self._act(m_mor, "Xóa vùng nhỏ (Chấm bẩn)", self.cmd_remove_small)

        # 7. BIÊN (M7)
        m_edg = mb.addMenu("Biên")
        self._act(m_edg, "Dò biên Sobel", self.cmd_sobel)
        self._act(m_edg, "Dò biên LoG", self.cmd_log_edge)
        self._act(m_edg, "Dò biên Canny", self.cmd_canny)
        self._act(m_edg, "Hiệu ứng phác thảo", self.cmd_sketch)
        self._act(m_edg, "Tìm đường thẳng (Hough)", self.cmd_hough_lines)

        # 8. VIDEO (M8)
        m_vid = mb.addMenu("Video")
        self._act(m_vid, "Mở Video Studio (Phát, Trích khung, Áp lệnh, Chuyển động)...", self.open_video)

        # 9. AI (Phương án A)
        m_ai = mb.addMenu("AI")
        self._act(m_ai, "Tô màu ảnh đen trắng tự động (Zhang 2016)", self.cmd_ai_colorize)

        # 10. TRỢ GIÚP
        m_help = mb.addMenu("Trợ giúp")
        self._act(m_help, "Hướng dẫn nhanh & Thông tin", self._show_about)

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # =========================================================================
    # TOOLBOX & DOCKS & STATUSBAR (G1–G4)
    # =========================================================================
    def _init_toolbar(self):
        tb = QToolBar("Hộp công cụ")
        tb.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, tb)

        self.act_pointer = QAction("Con trỏ", self, checkable=True, checked=True)
        self.act_crop = QAction("Cắt vùng", self, checkable=True)
        self.act_eyedrop = QAction("Ống hút", self, checkable=True)

        for act, name in [(self.act_pointer, "pointer"), (self.act_crop, "crop"), (self.act_eyedrop, "eyedropper")]:
            act.triggered.connect(lambda checked, n=name: self._select_tool(n))
            tb.addAction(act)

        tb.addSeparator()
        tb.addAction(QAction("Phóng +", self, triggered=lambda: self.canvas.set_zoom(self.canvas.get_zoom() * 1.25)))
        tb.addAction(QAction("Thu -", self, triggered=lambda: self.canvas.set_zoom(self.canvas.get_zoom() * 0.8)))
        tb.addAction(QAction("100%", self, triggered=lambda: self.canvas.set_zoom(1.0)))
        tb.addSeparator()
        tb.addAction(QAction("Trước/Sau", self, triggered=self.cmd_compare_original))

    def _select_tool(self, tool_name: str):
        self.act_pointer.setChecked(tool_name == "pointer")
        self.act_crop.setChecked(tool_name == "crop")
        self.act_eyedrop.setChecked(tool_name == "eyedropper")
        self.canvas.set_tool(tool_name)

    def _init_docks(self):
        self.hist_widget = HistogramWidget()
        dock_hist = QDockWidget("HISTOGRAM", self)
        dock_hist.setWidget(self.hist_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_hist)

        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.addWidget(QLabel("Chọn một lệnh có tham số để xem trước."))
        self.param_layout.addStretch()
        dock_param = QDockWidget("THAM SỐ (XEM TRƯỚC)", self)
        dock_param.setWidget(self.param_container)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_param)

        self.history_list = QListWidget()
        dock_ops = QDockWidget("LỊCH SỬ THAO TÁC", self)
        dock_ops.setWidget(self.history_list)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_ops)

        self.result_box = QTextEdit(readOnly=True)
        dock_res = QDockWidget("BẢNG KẾT QUẢ", self)
        dock_res.setWidget(self.result_box)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_res)

    def _init_statusbar(self):
        self.status_label = QLabel("Sẵn sàng — Nhấn Ctrl+O để mở ảnh.")
        self.statusBar().addWidget(self.status_label)

    def _update_status_pixel(self, x: int, y: int, val_str: str):
        if self.current_img is None:
            return
        info = m1_basic.get_image_info(self.current_img)
        zoom_pct = int(self.canvas.get_zoom() * 100)
        self.status_label.setText(
            f"x={x}, y={y}  |  {val_str}  |  {info['width']}×{info['height']}, "
            f"{info['channels']} kênh  |  Phóng {zoom_pct}%"
        )

    # =========================================================================
    # QUẢN LÝ LỊCH SỬ & PREVIEW (G2, G3)
    # =========================================================================
    def _push_history(self, name: str, new_img: np.ndarray, func=None, kwargs: dict | None = None):
        self.history_stack = self.history_stack[: self.history_index + 1]
        self.history_stack.append((name, new_img.copy(), func, kwargs or {}))
        if len(self.history_stack) > self.max_history:
            self.history_stack.pop(0)
        self.history_index = len(self.history_stack) - 1
        self.current_img = new_img.copy()
        self._refresh_ui()

    def _refresh_ui(self):
        self.canvas.set_image(self.current_img)
        self.hist_widget.update_histogram(self.current_img)
        self.history_list.clear()
        for idx, (name, _, _, _) in enumerate(self.history_stack):
            mark = "▶ " if idx == self.history_index else "   "
            self.history_list.addItem(f"{mark}{idx + 1}. {name}")
        self.history_list.scrollToBottom()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_img = self.history_stack[self.history_index][1].copy()
            self._refresh_ui()

    def redo(self):
        if self.history_index < len(self.history_stack) - 1:
            self.history_index += 1
            self.current_img = self.history_stack[self.history_index][1].copy()
            self._refresh_ui()

    def reset_original(self):
        if self.original_img is not None:
            self._push_history("Khôi phục ảnh gốc", self.original_img)

    def _clear_param_panel(self):
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_sub_layout(item.layout())

    def _clear_sub_layout(self, layout):
        while layout.count():
            sub = layout.takeAt(0)
            if sub.widget():
                sub.widget().deleteLater()
            elif sub.layout():
                self._clear_sub_layout(sub.layout())

    def _add_preview_controls(self, title: str, get_apply_info, extra_btn=None):
        """Tạo khung nút Giữ so sánh, Áp dụng, Hủy dùng chung cho mọi lệnh có tham số (G2)."""
        btn_hold = QPushButton("Giữ chuột để xem ảnh trước", objectName="secondaryBtn")
        btn_hold.pressed.connect(lambda: self.canvas.set_image(self.current_img))
        btn_hold.released.connect(lambda: self.canvas.set_image(self.preview_img))

        btn_side = QPushButton("So sánh Trước/Sau cạnh nhau", objectName="secondaryBtn")
        btn_side.clicked.connect(
            lambda: MultiImageDialog(
                f"So sánh Trước / Sau — {title}",
                [("Ảnh trước", self.current_img), ("Ảnh sau (Xem trước)", self.preview_img)],
                self
            ).exec() if self.preview_img is not None else None
        )

        btn_apply = QPushButton("Áp dụng", objectName="accentBtn")
        btn_cancel = QPushButton("Hủy", objectName="secondaryBtn")

        def on_apply():
            if self.preview_img is None:
                return

            label, func, kwargs, log_msg = get_apply_info()
            self._push_history(label, self.preview_img, func, kwargs)

            if log_msg:
                self.result_box.append(log_msg)

            # Giữ ảnh xem trước cùng các điều khiển đang mở
            self.preview_img = self.current_img.copy()

        def on_cancel():
            self.preview_img = None
            self._refresh_ui()
            self._clear_param_panel()
            self.param_layout.addWidget(QLabel("Đã hủy thao tác."))
            self.param_layout.addStretch()

        btn_apply.clicked.connect(on_apply)
        btn_cancel.clicked.connect(on_cancel)

        self.param_layout.addWidget(btn_hold)
        self.param_layout.addWidget(btn_side)
        if extra_btn is not None:
            self.param_layout.addWidget(extra_btn)
        row = QHBoxLayout()
        row.addWidget(btn_apply)
        row.addWidget(btn_cancel)
        self.param_layout.addLayout(row)
        self.param_layout.addStretch()

    def _require_image(self) -> bool:
        if self.current_img is None:
            QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng mở một ảnh trước (Ctrl+O).")
            return False
        return True

    # =========================================================================
    # MENU TỆP (G4)
    # =========================================================================
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            QMessageBox.critical(self, "Lỗi tệp", "Không thể mở tệp ảnh! Định dạng không hợp lệ.")
            return

        self.current_filepath = path
        self.original_img = img.copy()
        self.clean_img_for_psnr = img.copy()
        self.history_stack.clear()
        self.history_index = -1
        self._push_history("Mở ảnh gốc", img)
        self.result_box.append(f"Đã mở ảnh: {os.path.basename(path)}")

    def save_image(self):
        if not self._require_image():
            return
        if not self.current_filepath:
            self.save_image_as()
            return
        self._write_image(self.current_filepath)

    def save_image_as(self):
        if not self._require_image():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh thành", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.current_filepath = path
            self._write_image(path)

    def _write_image(self, path: str):
        ext = ".png" if path.lower().endswith(".png") else ".jpg"
        ok, buf = cv2.imencode(ext, self.current_img)
        if ok:
            buf.tofile(path)
            self.result_box.append(f"Đã lưu ảnh: {path}")

    def cmd_compare_original(self):
        if not self._require_image():
            return
        MultiImageDialog(
            "So sánh Ảnh gốc & Ảnh hiện tại",
            [("Ảnh gốc", self.original_img), ("Ảnh hiện tại", self.current_img)],
            self
        ).exec()

    # =========================================================================
    # M1: ẢNH CƠ BẢN
    # =========================================================================
    def cmd_image_info(self):
        if not self._require_image():
            return
        info = m1_basic.get_image_info(self.current_img)
        msg = (
            f"Chiều rộng: {info['width']} px\n"
            f"Chiều cao: {info['height']} px\n"
            f"Số kênh màu: {info['channels']}\n"
            f"Kiểu dữ liệu: {info['dtype']}\n"
            f"Dung lượng chưa nén: {info['size_bytes']:,} bytes"
        )
        QMessageBox.information(self, "Thông tin ảnh", msg)

    def cmd_grayscale(self):
        if not self._require_image():
            return
        out = m1_basic.to_grayscale(self.current_img)
        self._push_history("Chuyển ảnh xám", out, m1_basic.to_grayscale, {})

    def _on_eyedropper_picked(self, x: int, y: int, val_str: str):
        self.result_box.append(f"[Ống hút màu] ({x}, {y}) -> {val_str}")

    def _on_crop_selected(self, x1: int, y1: int, x2: int, y2: int):
        if self.current_img is None:
            return
        cropped = m1_basic.crop_image(self.current_img, x1, y1, x2, y2)
        self._push_history(f"Cắt ảnh ({x1},{y1})-({x2},{y2})", cropped)
        self._select_tool("pointer")

    def cmd_resize_dialog(self):
        if not self._require_image():
            return
        h, w = self.current_img.shape[:2]
        dlg = QDialog(self)
        dlg.setWindowTitle("Đổi kích thước ảnh")
        form = QFormLayout(dlg)

        sp_w = QSpinBox(minimum=1, maximum=10000, value=w)
        sp_h = QSpinBox(minimum=1, maximum=10000, value=h)
        sp_pct = QSpinBox(minimum=5, maximum=400, value=100, suffix="%")
        cb_ratio = QCheckBox("Giữ tỷ lệ gốc", checked=True)

        def on_w(val):
            if cb_ratio.isChecked():
                sp_h.blockSignals(True)
                sp_h.setValue(max(1, int(val * h / w)))
                sp_h.blockSignals(False)

        def on_pct(val):
            sp_w.blockSignals(True)
            sp_h.blockSignals(True)
            sp_w.setValue(max(1, int(w * val / 100)))
            sp_h.setValue(max(1, int(h * val / 100)))
            sp_w.blockSignals(False)
            sp_h.blockSignals(False)

        sp_w.valueChanged.connect(on_w)
        sp_pct.valueChanged.connect(on_pct)

        form.addRow("Tỷ lệ (%):", sp_pct)
        form.addRow("Chiều rộng (px):", sp_w)
        form.addRow("Chiều cao (px):", sp_h)
        form.addRow(cb_ratio)
        form.addRow(QPushButton("Áp dụng", clicked=dlg.accept))

        if dlg.exec() == QDialog.Accepted:
            nw, nh = sp_w.value(), sp_h.value()
            out = m1_basic.resize_image(self.current_img, nw, nh)
            self._push_history(f"Đổi kích thước {nw}×{nh}", out, m1_basic.resize_image, {"new_w": nw, "new_h": nh})

    def cmd_quantize(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        levels_map = [2, 4, 8, 16, 32, 64, 128]
        lbl = QLabel("Số mức xám L: 16")
        sl = QSlider(Qt.Horizontal, minimum=0, maximum=len(levels_map) - 1, value=3)

        def update():
            L = levels_map[sl.value()]
            lbl.setText(f"Số mức xám L: {L}")
            self.preview_img = m1_basic.quantize_gray_levels(self.current_img, L)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Giảm mức xám</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls(
            "Giảm mức xám",
            lambda: (f"Giảm mức xám L={levels_map[sl.value()]}", m1_basic.quantize_gray_levels, {"levels": levels_map[sl.value()]}, "")
        )
        update()

    # =========================================================================
    # M2: ĐIỀU CHỈNH ĐỘ SÁNG, TƯƠNG PHẢN & HISTOGRAM
    # =========================================================================
    def cmd_negative(self):
        if not self._require_image():
            return
        out = m2_adjust.negative(self.current_img)
        self._push_history("Âm bản", out, m2_adjust.negative, {})

    def cmd_log(self):
        if not self._require_image():
            return
        out = m2_adjust.log_transform(self.current_img)
        self._push_history("Biến đổi Log", out, m2_adjust.log_transform, {})

    def cmd_gamma(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Gamma: 1.00")
        sl = QSlider(Qt.Horizontal, minimum=10, maximum=300, value=100)

        def update():
            g = sl.value() / 100.0
            lbl.setText(f"Gamma: {g:.2f}")
            self.preview_img = m2_adjust.gamma_transform(self.current_img, g)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Biến đổi Gamma [TC]</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls(
            "Gamma",
            lambda: (f"Gamma={sl.value()/100:.2f}", m2_adjust.gamma_transform, {"gamma": sl.value() / 100.0}, "")
        )
        update()

    def cmd_contrast_stretch(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_a = QLabel("Mức đen a: 20")
        sl_a = QSlider(Qt.Horizontal, minimum=0, maximum=254, value=20)
        lbl_b = QLabel("Mức trắng b: 235")
        sl_b = QSlider(Qt.Horizontal, minimum=1, maximum=255, value=235)

        def update():
            a, b = sl_a.value(), sl_b.value()
            lbl_a.setText(f"Mức đen a: {a}")
            lbl_b.setText(f"Mức trắng b: {b}")
            self.preview_img = m2_adjust.contrast_stretch(self.current_img, a, b)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl_a.valueChanged.connect(update)
        sl_b.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Kéo giãn tương phản</b>"))
        for w in (lbl_a, sl_a, lbl_b, sl_b):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Kéo giãn tương phản",
            lambda: (f"Kéo giãn [{sl_a.value()}, {sl_b.value()}]", m2_adjust.contrast_stretch, {"low": sl_a.value(), "high": sl_b.value()}, "")
        )
        update()

    def cmd_hist_eq(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        self.preview_img = m2_adjust.histogram_equalize_tc(self.current_img)
        self.canvas.set_image(self.preview_img)
        self.hist_widget.update_histogram(self.preview_img)

        btn_cmp_cv = QPushButton("So sánh với OpenCV (cv2.equalizeHist)")
        btn_cmp_cv.clicked.connect(
            lambda: MultiImageDialog(
                "So sánh Cân bằng Histogram: Tự cài [TC] vs OpenCV",
                [
                    ("Tự cài [TC] (NumPy)", self.preview_img),
                    ("OpenCV (cv2.equalizeHist)", m2_adjust.histogram_equalize_opencv(self.current_img)),
                ],
                self
            ).exec()
        )
        self.param_layout.addWidget(QLabel("<b>Cân bằng Histogram [TC]</b>"))
        self._add_preview_controls(
            "Cân bằng Histogram [TC]",
            lambda: ("Cân bằng Histogram [TC]", m2_adjust.histogram_equalize_tc, {}, ""),
            extra_btn=btn_cmp_cv
        )

    def cmd_clahe(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_c = QLabel("Clip Limit: 2.0")
        sl_c = QSlider(Qt.Horizontal, minimum=10, maximum=50, value=20)
        lbl_t = QLabel("Kích thước lưới ô: 8")
        sl_t = QSlider(Qt.Horizontal, minimum=4, maximum=16, value=8)

        def update():
            c, t = sl_c.value() / 10.0, sl_t.value()
            lbl_c.setText(f"Clip Limit: {c:.1f}")
            lbl_t.setText(f"Kích thước lưới ô: {t}")
            self.preview_img = m2_adjust.apply_clahe(self.current_img, c, t)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl_c.valueChanged.connect(update)
        sl_t.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>CLAHE</b>"))
        for w in (lbl_c, sl_c, lbl_t, sl_t):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "CLAHE",
            lambda: (f"CLAHE (clip={sl_c.value()/10:.1f}, grid={sl_t.value()})", m2_adjust.apply_clahe, {"clip_limit": sl_c.value() / 10.0, "tile_size": sl_t.value()}, "")
        )
        update()

    # =========================================================================
    # M3: BỘ LỌC LÀM MỊN & LÀM SẮC NÉT
    # =========================================================================
    def cmd_mean_blur(self):
        self._simple_kernel_filter("Lọc Trung bình", m3_filter.mean_blur)

    def cmd_median_blur(self):
        self._simple_kernel_filter("Lọc Trung vị", m3_filter.median_blur)

    def _simple_kernel_filter(self, title: str, func):
        if not self._require_image():
            return
        self._clear_param_panel()
        sizes = [3, 5, 7, 9]
        lbl = QLabel("Kích thước cửa sổ: 3×3")
        sl = QSlider(Qt.Horizontal, minimum=0, maximum=3, value=0)

        def update():
            k = sizes[sl.value()]
            lbl.setText(f"Kích thước cửa sổ: {k}×{k}")
            self.preview_img = func(self.current_img, k)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel(f"<b>{title}</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls(title, lambda: (f"{title} ({sizes[sl.value()]}×{sizes[sl.value()]})", func, {"ksize": sizes[sl.value()]}, ""))
        update()

    def cmd_gaussian_blur(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        sizes = [3, 5, 7, 9, 11]
        lbl_k = QLabel("Cửa sổ: 5×5")
        sl_k = QSlider(Qt.Horizontal, minimum=0, maximum=4, value=1)
        lbl_s = QLabel("Sigma: 1.5")
        sl_s = QSlider(Qt.Horizontal, minimum=5, maximum=50, value=15)

        def update():
            k, s = sizes[sl_k.value()], sl_s.value() / 10.0
            lbl_k.setText(f"Cửa sổ: {k}×{k}")
            lbl_s.setText(f"Sigma: {s:.1f}")
            self.preview_img = m3_filter.gaussian_blur(self.current_img, k, s)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl_k.valueChanged.connect(update)
        sl_s.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Lọc Gaussian</b>"))
        for w in (lbl_k, sl_k, lbl_s, sl_s):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Lọc Gaussian",
            lambda: (f"Gaussian (k={sizes[sl_k.value()]}, σ={sl_s.value()/10:.1f})", m3_filter.gaussian_blur, {"ksize": sizes[sl_k.value()], "sigma": sl_s.value() / 10.0}, "")
        )
        update()

    def cmd_laplacian_sharpen(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Hệ số c: 0.50")
        sl = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)

        def update():
            c = sl.value() / 100.0
            lbl.setText(f"Hệ số c: {c:.2f}")
            self.preview_img = m3_filter.sharpen_laplacian(self.current_img, c)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Sắc nét Laplacian</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls("Sắc nét Laplacian", lambda: (f"Laplacian c={sl.value()/100:.2f}", m3_filter.sharpen_laplacian, {"c": sl.value() / 100.0}, ""))
        update()

    def cmd_unsharp(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Hệ số k: 1.50")
        sl = QSlider(Qt.Horizontal, minimum=0, maximum=300, value=150)

        def update():
            k = sl.value() / 100.0
            lbl.setText(f"Hệ số k: {k:.2f}")
            self.preview_img = m3_filter.unsharp_masking(self.current_img, k)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Unsharp Masking</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls("Unsharp Masking", lambda: (f"Unsharp k={sl.value()/100:.2f}", m3_filter.unsharp_masking, {"k": sl.value() / 100.0}, ""))
        update()

    # =========================================================================
    # M4: NHIỄU, GIẢM NHIỄU [TC] & PSNR
    # =========================================================================
    def cmd_add_gaussian_noise(self):
        if not self._require_image():
            return
        self.clean_img_for_psnr = self.current_img.copy()
        self._clear_param_panel()
        lbl = QLabel("Độ lệch chuẩn σ: 20")
        sl = QSlider(Qt.Horizontal, minimum=5, maximum=50, value=20)

        def update():
            s = sl.value()
            lbl.setText(f"Độ lệch chuẩn σ: {s}")
            self.preview_img = m4_noise.add_gaussian_noise(self.current_img, s)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Thêm nhiễu Gaussian</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls(
            "Nhiễu Gaussian",
            lambda: (
                f"Thêm nhiễu Gaussian σ={sl.value()}", m4_noise.add_gaussian_noise, {"sigma": sl.value()},
                f"[M4] Đã thêm nhiễu Gaussian (σ={sl.value()}) | PSNR = {m4_noise.compute_psnr(self.clean_img_for_psnr, self.preview_img):.2f} dB"
            )
        )
        update()

    def cmd_add_sp_noise(self):
        if not self._require_image():
            return
        self.clean_img_for_psnr = self.current_img.copy()
        self._clear_param_panel()
        lbl = QLabel("Tỷ lệ nhiễu p: 5%")
        sl = QSlider(Qt.Horizontal, minimum=1, maximum=20, value=5)

        def update():
            p = sl.value()
            lbl.setText(f"Tỷ lệ nhiễu p: {p}%")
            self.preview_img = m4_noise.add_salt_pepper_noise(self.current_img, p)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Thêm nhiễu muối tiêu</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls(
            "Nhiễu muối tiêu",
            lambda: (
                f"Thêm nhiễu muối tiêu {sl.value()}%", m4_noise.add_salt_pepper_noise, {"percent": sl.value()},
                f"[M4] Đã thêm nhiễu muối tiêu ({sl.value()}%) | PSNR = {m4_noise.compute_psnr(self.clean_img_for_psnr, self.preview_img):.2f} dB"
            )
        )
        update()

    def cmd_denoise(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        cb = QComboBox()
        cb.addItems([
            "Lọc Trung bình (3×3)",
            "Lọc Trung vị (3×3)",
            "Lọc Trung bình hình học [TC] (3×3)",
            "Lọc Phản điều hòa [TC] (3×3)",
        ])
        lbl_q = QLabel("Bậc Q (Phản điều hòa): 0.0")
        sl_q = QSlider(Qt.Horizontal, minimum=-20, maximum=20, value=0)
        lbl_psnr = QLabel("PSNR: -- dB")
        lbl_psnr.setStyleSheet("color: #4ade80; font-weight: bold;")

        def run_filter():
            idx = cb.currentIndex()
            q = sl_q.value() / 10.0
            lbl_q.setVisible(idx == 3)
            sl_q.setVisible(idx == 3)
            lbl_q.setText(f"Bậc Q (Phản điều hòa): {q:.1f}")

            if idx == 0:
                self.preview_img = m3_filter.mean_blur(self.current_img, 3)
            elif idx == 1:
                self.preview_img = m3_filter.median_blur(self.current_img, 3)
            elif idx == 2:
                self.preview_img = m4_noise.geometric_mean_filter_tc(self.current_img, 3)
            else:
                self.preview_img = m4_noise.contraharmonic_filter_tc(self.current_img, 3, q)

            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)
            if self.clean_img_for_psnr is not None and self.clean_img_for_psnr.shape == self.preview_img.shape:
                psnr = m4_noise.compute_psnr(self.clean_img_for_psnr, self.preview_img)
                lbl_psnr.setText(f"PSNR so với ảnh sạch: {psnr:.2f} dB")
            else:
                lbl_psnr.setText("PSNR: (chưa có ảnh gốc sạch cùng kích thước)")

        cb.currentIndexChanged.connect(run_filter)
        sl_q.valueChanged.connect(run_filter)

        self.param_layout.addWidget(QLabel("<b>Giảm nhiễu & Đo PSNR (M4)</b>"))
        for w in (cb, lbl_q, sl_q, lbl_psnr):
            self.param_layout.addWidget(w)

        def get_info():
            idx = cb.currentIndex()
            q = sl_q.value() / 10.0
            funcs = [
                (m3_filter.mean_blur, {"ksize": 3}),
                (m3_filter.median_blur, {"ksize": 3}),
                (m4_noise.geometric_mean_filter_tc, {"ksize": 3}),
                (m4_noise.contraharmonic_filter_tc, {"ksize": 3, "q": q}),
            ]
            f, kw = funcs[idx]
            name = cb.currentText() + (f" Q={q:.1f}" if idx == 3 else "")
            return name, f, kw, f"[M4] {name} -> {lbl_psnr.text()}"

        self._add_preview_controls("Giảm nhiễu", get_info)
        run_filter()

    # =========================================================================
    # M5: CHỈNH MÀU, CÂN BẰNG MÀU & TÔ MÀU GIẢ
    # =========================================================================
    def cmd_view_hsv(self):
        if not self._require_image():
            return
        h, s, v = m5_color.split_hsv_channels(self.current_img)
        MultiImageDialog("3 Kênh không gian màu HSV (M5)", [("Kênh H (Hue)", h), ("Kênh S (Saturation)", s), ("Kênh V (Value)", v)], self).exec()

    def cmd_adjust_hsv(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_h = QLabel("Dịch sắc độ H: 0")
        sl_h = QSlider(Qt.Horizontal, minimum=-90, maximum=90, value=0)
        lbl_s = QLabel("Bão hòa S: 1.00×")
        sl_s = QSlider(Qt.Horizontal, minimum=0, maximum=200, value=100)
        lbl_v = QLabel("Độ sáng V: 1.00×")
        sl_v = QSlider(Qt.Horizontal, minimum=0, maximum=200, value=100)

        def update():
            h, s, v = sl_h.value(), sl_s.value() / 100.0, sl_v.value() / 100.0
            lbl_h.setText(f"Dịch sắc độ H: {h:+d}")
            lbl_s.setText(f"Bão hòa S: {s:.2f}×")
            lbl_v.setText(f"Độ sáng V: {v:.2f}×")
            self.preview_img = m5_color.adjust_hsv(self.current_img, h, s, v)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        for sl in (sl_h, sl_s, sl_v):
            sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Sắc độ / Bão hòa / Độ sáng (HSV)</b>"))
        for w in (lbl_h, sl_h, lbl_s, sl_s, lbl_v, sl_v):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Chỉnh HSV",
            lambda: (f"HSV (H={sl_h.value():+d}, S={sl_s.value()/100:.2f}, V={sl_v.value()/100:.2f})",
                     m5_color.adjust_hsv, {"h_shift": sl_h.value(), "s_scale": sl_s.value() / 100.0, "v_scale": sl_v.value() / 100.0}, "")
        )
        update()

    def cmd_color_balance(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_b = QLabel("Hệ số B (Xanh dương): 1.00")
        sl_b = QSlider(Qt.Horizontal, minimum=50, maximum=150, value=100)
        lbl_g = QLabel("Hệ số G (Xanh lá): 1.00")
        sl_g = QSlider(Qt.Horizontal, minimum=50, maximum=150, value=100)
        lbl_r = QLabel("Hệ số R (Đỏ): 1.00")
        sl_r = QSlider(Qt.Horizontal, minimum=50, maximum=150, value=100)

        def update():
            b, g, r = sl_b.value() / 100.0, sl_g.value() / 100.0, sl_r.value() / 100.0
            lbl_b.setText(f"Hệ số B (Xanh dương): {b:.2f}")
            lbl_g.setText(f"Hệ số G (Xanh lá): {g:.2f}")
            lbl_r.setText(f"Hệ số R (Đỏ): {r:.2f}")
            self.preview_img = m5_color.color_balance_bgr(self.current_img, b, g, r)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        for sl in (sl_b, sl_g, sl_r):
            sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Cân bằng màu (Khử ám vàng)</b>"))
        for w in (lbl_b, sl_b, lbl_g, sl_g, lbl_r, sl_r):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Cân bằng màu",
            lambda: (f"Cân bằng màu (B={sl_b.value()/100:.2f}, G={sl_g.value()/100:.2f}, R={sl_r.value()/100:.2f})",
                     m5_color.color_balance_bgr, {"b_gain": sl_b.value() / 100.0, "g_gain": sl_g.value() / 100.0, "r_gain": sl_r.value() / 100.0}, "")
        )
        update()

    def cmd_pseudocolor(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        cb = QComboBox()
        cb.addItems(list(m5_color.COLORMAPS.keys()))

        def update():
            self.preview_img = m5_color.apply_pseudocolor(self.current_img, cb.currentText())
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        cb.currentIndexChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Tô màu giả (Colormap)</b>"))
        self.param_layout.addWidget(cb)
        self._add_preview_controls("Tô màu giả", lambda: (f"Tô màu giả ({cb.currentText()})", m5_color.apply_pseudocolor, {"cmap_name": cb.currentText()}, ""))
        update()

    # =========================================================================
    # M6: HÌNH THÁI HỌC
    # =========================================================================
    def cmd_binarize(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Ngưỡng: 127")
        sl = QSlider(Qt.Horizontal, minimum=0, maximum=255, value=127)
        cb_otsu = QCheckBox("Tự động chọn ngưỡng Otsu")
        cb_inv = QCheckBox("Đảo màu (Chữ/Nét thành trắng)")

        def update():
            sl.setEnabled(not cb_otsu.isChecked())
            lbl.setText("Ngưỡng: Otsu (Tự động)" if cb_otsu.isChecked() else f"Ngưỡng: {sl.value()}")
            self.preview_img = m6_morph.binarize(self.current_img, sl.value(), cb_otsu.isChecked(), cb_inv.isChecked())
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        cb_otsu.stateChanged.connect(update)
        cb_inv.stateChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Nhị phân hóa (M6)</b>"))
        for w in (lbl, sl, cb_otsu, cb_inv):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Nhị phân hóa",
            lambda: ("Nhị phân hóa (Otsu)" if cb_otsu.isChecked() else f"Nhị phân hóa T={sl.value()}",
                     m6_morph.binarize, {"thresh": sl.value(), "use_otsu": cb_otsu.isChecked(), "invert": cb_inv.isChecked()}, "")
        )
        update()

    def cmd_morph(self, op: str, title: str):
        if not self._require_image():
            return
        self._clear_param_panel()
        cb_shape = QComboBox()
        cb_shape.addItems(list(m6_morph.SHAPES.keys()))
        lbl_k = QLabel("Kích thước phần tử cấu trúc: 3")
        sl_k = QSlider(Qt.Horizontal, minimum=1, maximum=7, value=1)  # 2*v + 1 -> 3..15
        lbl_i = QLabel("Số lần lặp: 1")
        sl_i = QSlider(Qt.Horizontal, minimum=1, maximum=5, value=1)

        def update():
            k = sl_k.value() * 2 + 1
            it = sl_i.value()
            lbl_k.setText(f"Kích thước phần tử cấu trúc: {k}×{k}")
            lbl_i.setText(f"Số lần lặp: {it}")
            self.preview_img = m6_morph.morphology_op(self.current_img, op, cb_shape.currentText(), k, it)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        cb_shape.currentIndexChanged.connect(update)
        sl_k.valueChanged.connect(update)
        sl_i.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel(f"<b>{title}</b>"))
        for w in (cb_shape, lbl_k, sl_k, lbl_i, sl_i):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            title,
            lambda: (f"{title} (k={sl_k.value()*2+1}, it={sl_i.value()})",
                     m6_morph.morphology_op, {"op": op, "shape_name": cb_shape.currentText(), "ksize": sl_k.value() * 2 + 1, "iterations": sl_i.value()}, "")
        )
        update()

    def cmd_boundary(self):
        if not self._require_image():
            return
        out = m6_morph.extract_boundary(self.current_img, 3)
        self._push_history("Trích biên hình thái", out, m6_morph.extract_boundary, {"ksize": 3})

    def cmd_remove_small(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Diện tích tối thiểu (px): 50")
        sl = QSlider(Qt.Horizontal, minimum=5, maximum=500, value=50)
        lbl_cnt = QLabel("Số vùng đã xóa: 0")
        lbl_cnt.setStyleSheet("color: #facc15; font-weight: bold;")
        state = {"removed": 0}

        def update():
            area = sl.value()
            lbl.setText(f"Diện tích tối thiểu (px): {area}")
            cleaned, cnt = m6_morph.remove_small_regions(self.current_img, area)
            state["removed"] = cnt
            lbl_cnt.setText(f"Số vùng nhỏ đã xóa: {cnt}")
            self.preview_img = cleaned
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Xóa vùng nhỏ / Chấm bẩn (M6)</b>"))
        for w in (lbl, sl, lbl_cnt):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Xóa vùng nhỏ",
            lambda: (f"Xóa vùng nhỏ (<{sl.value()}px)", m6_morph.remove_small_regions, {"min_area": sl.value()},
                     f"[M6] Đã xóa {state['removed']} vùng nhiễu có diện tích < {sl.value()} px")
        )
        update()

    # =========================================================================
    # M7: BIÊN, PHÁC THẢO & HOUGH
    # =========================================================================
    def cmd_sobel(self):
        if not self._require_image():
            return
        out = m7_edge.sobel_edge(self.current_img)
        self._push_history("Dò biên Sobel", out, m7_edge.sobel_edge, {})

    def cmd_log_edge(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl = QLabel("Sigma Gaussian: 1.0")
        sl = QSlider(Qt.Horizontal, minimum=5, maximum=50, value=10)

        def update():
            s = sl.value() / 10.0
            lbl.setText(f"Sigma Gaussian: {s:.1f}")
            self.preview_img = m7_edge.log_edge(self.current_img, s)
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Dò biên LoG</b>"))
        self.param_layout.addWidget(lbl)
        self.param_layout.addWidget(sl)
        self._add_preview_controls("Dò biên LoG", lambda: (f"LoG σ={sl.value()/10:.1f}", m7_edge.log_edge, {"sigma": sl.value() / 10.0}, ""))
        update()

    def cmd_canny(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_l = QLabel("Ngưỡng thấp: 50")
        sl_l = QSlider(Qt.Horizontal, minimum=0, maximum=250, value=50)
        lbl_h = QLabel("Ngưỡng cao: 150")
        sl_h = QSlider(Qt.Horizontal, minimum=10, maximum=300, value=150)

        def update():
            lbl_l.setText(f"Ngưỡng thấp: {sl_l.value()}")
            lbl_h.setText(f"Ngưỡng cao: {sl_h.value()}")
            self.preview_img = m7_edge.canny_edge(self.current_img, sl_l.value(), sl_h.value())
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl_l.valueChanged.connect(update)
        sl_h.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Dò biên Canny</b>"))
        for w in (lbl_l, sl_l, lbl_h, sl_h):
            self.param_layout.addWidget(w)
        self._add_preview_controls("Canny", lambda: (f"Canny ({sl_l.value()}, {sl_h.value()})", m7_edge.canny_edge, {"low": sl_l.value(), "high": sl_h.value()}, ""))
        update()

    def cmd_sketch(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_l = QLabel("Ngưỡng thấp: 50")
        sl_l = QSlider(Qt.Horizontal, minimum=0, maximum=250, value=50)
        lbl_h = QLabel("Ngưỡng cao: 150")
        sl_h = QSlider(Qt.Horizontal, minimum=10, maximum=300, value=150)
        cb_soft = QCheckBox("Làm mềm nét (Gaussian nhẹ)", checked=True)

        def update():
            lbl_l.setText(f"Ngưỡng thấp: {sl_l.value()}")
            lbl_h.setText(f"Ngưỡng cao: {sl_h.value()}")
            self.preview_img = m7_edge.sketch_effect(self.current_img, sl_l.value(), sl_h.value(), cb_soft.isChecked())
            self.canvas.set_image(self.preview_img)
            self.hist_widget.update_histogram(self.preview_img)

        sl_l.valueChanged.connect(update)
        sl_h.valueChanged.connect(update)
        cb_soft.stateChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Hiệu ứng Phác thảo</b>"))
        for w in (lbl_l, sl_l, lbl_h, sl_h, cb_soft):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Hiệu ứng phác thảo",
            lambda: ("Hiệu ứng phác thảo", m7_edge.sketch_effect, {"low": sl_l.value(), "high": sl_h.value(), "soft_blur": cb_soft.isChecked()}, "")
        )
        update()

    def cmd_hough_lines(self):
        if not self._require_image():
            return
        self._clear_param_panel()
        lbl_t = QLabel("Threshold (Số phiếu): 80")
        sl_t = QSlider(Qt.Horizontal, minimum=10, maximum=250, value=80)
        lbl_m = QLabel("Min Line Length: 50")
        sl_m = QSlider(Qt.Horizontal, minimum=10, maximum=300, value=50)
        lbl_g = QLabel("Max Line Gap: 10")
        sl_g = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=10)
        lbl_info = QLabel("Kết quả: --")
        lbl_info.setStyleSheet("color: #38bdf8; font-weight: bold;")
        state = {"msg": ""}

        def update():
            t, m, g = sl_t.value(), sl_m.value(), sl_g.value()
            lbl_t.setText(f"Threshold (Số phiếu): {t}")
            lbl_m.setText(f"Min Line Length: {m}")
            lbl_g.setText(f"Max Line Gap: {g}")
            out, count, angle = m7_edge.detect_lines_hough(self.current_img, t, m, g)
            self.preview_img = out
            self.canvas.set_image(out)
            self.hist_widget.update_histogram(out)
            if angle is not None:
                state["msg"] = f"Tìm thấy {count} đoạn | Góc đoạn dài nhất: {angle:.2f}°"
            else:
                state["msg"] = "Không tìm thấy đoạn thẳng nào."
            lbl_info.setText(state["msg"])

        for sl in (sl_t, sl_m, sl_g):
            sl.valueChanged.connect(update)
        self.param_layout.addWidget(QLabel("<b>Tìm đường thẳng (HoughLinesP)</b>"))
        for w in (lbl_t, sl_t, lbl_m, sl_m, lbl_g, sl_g, lbl_info):
            self.param_layout.addWidget(w)
        self._add_preview_controls(
            "Tìm đường thẳng Hough",
            lambda: ("Tìm đường thẳng Hough", m7_edge.detect_lines_hough,
                     {"threshold": sl_t.value(), "min_length": sl_m.value(), "max_gap": sl_g.value()},
                     f"[M7 Hough] {state['msg']}")
        )
        update()

    # =========================================================================
    # M8: VIDEO STUDIO
    # =========================================================================
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        replayable_ops = [
            (name, func, kwargs)
            for (name, _, func, kwargs) in self.history_stack[: self.history_index + 1]
            if func is not None
        ]
        try:
            dlg = VideoStudioDialog(path, replayable_ops, self)
            dlg.frame_extracted.connect(self._on_video_frame_extracted)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi mở video", str(e))

    def _on_video_frame_extracted(self, frame: np.ndarray, label: str):
        self.original_img = frame.copy()
        self.clean_img_for_psnr = frame.copy()
        self.history_stack.clear()
        self.history_index = -1
        self._push_history(label, frame)

    # =========================================================================
    # AI (PHƯƠNG ÁN A): TÔ MÀU TỰ ĐỘNG ẢNH ĐEN TRẮNG
    # =========================================================================
    def cmd_ai_colorize(self):
        if not self._require_image():
            return
        ready, missing = self.colorizer.is_ready()
        if not ready:
            QMessageBox.warning(
                self, "Thiếu tệp mô hình AI",
                "Chưa tìm thấy đủ 3 tệp mô hình Zhang et al. (2016) trong thư mục models/:\n"
                + "\n".join(f" • {m}" for m in missing)
                + "\n\nHãy chạy lệnh sau trong Terminal để tải tự động:\npython download_models.py"
            )
            return

        try:
            self.status_label.setText("Đang chạy mô hình AI tô màu ảnh đen trắng...")
            colorized = self.colorizer.colorize(self.current_img)
            self._push_history("AI Tô màu ảnh đen trắng", colorized, self.colorizer.colorize, {})
            self.result_box.append("[AI-A] Đã tô màu tự động thành công! Bạn có thể dùng Menu Màu > Sắc độ/Bão hòa (M5) để tinh chỉnh thêm.")
            self.status_label.setText("Hoàn tất tô màu AI.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi AI", f"Không thể chạy mô hình tô màu:\n{e}")

    def _show_about(self):
        QMessageBox.information(
            self, "Giới thiệu PhotoRestore",
            "PhotoRestore — Đề tài 3 (ARIN03020)\n"
            "• Giao diện: PySide6 (Dark Mode)\n"
            "• Đầy đủ chức năng G1–G4, M1–M8\n"
            "• Nâng cao: Phương án A — Tô màu ảnh đen trắng (Zhang et al., 2016)\n\n"
            "Mẹo: Giữ Ctrl + lăn chuột để Phóng to / Thu nhỏ ảnh."
        )
