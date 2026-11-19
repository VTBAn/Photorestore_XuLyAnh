DARK_THEME_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #1e1e1e;
    color: #e4e4e7;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #252526;
    border-bottom: 1px solid #3c3c3c;
    padding: 3px 6px;
}
QMenuBar::item {
    padding: 5px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #094771;
}
QMenu {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 3px;
}
QMenu::item:selected {
    background-color: #094771;
}
QToolBar {
    background-color: #252526;
    border-right: 1px solid #3c3c3c;
    spacing: 6px;
    padding: 6px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 6px 8px;
    color: #e4e4e7;
    font-weight: 500;
}
QToolButton:hover {
    background-color: #3a3d41;
}
QToolButton:checked {
    background-color: #094771;
    border: 1px solid #007acc;
}
QDockWidget {
    font-weight: 600;
}
QDockWidget::title {
    background-color: #2d2d30;
    padding: 7px 10px;
    border-bottom: 1px solid #3c3c3c;
    color: #60a5fa;
}
QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1177bb;
}
QPushButton:pressed {
    background-color: #094771;
}
QPushButton:disabled {
    background-color: #3a3d41;
    color: #777777;
}
QPushButton#secondaryBtn {
    background-color: #3c3c3c;
}
QPushButton#secondaryBtn:hover {
    background-color: #4f4f4f;
}
QPushButton#accentBtn {
    background-color: #15803d;
}
QPushButton#accentBtn:hover {
    background-color: #16a34a;
}
QListWidget, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px;
    color: #e4e4e7;
}
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-weight: 500;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #3c3c3c;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QProgressBar {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    text-align: center;
    background-color: #252526;
}
QProgressBar::chunk {
    background-color: #0e639c;
}
"""
