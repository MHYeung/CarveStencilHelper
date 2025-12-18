import os
import shutil

from PySide6.QtCore import Qt, QRect, Signal, QPoint, QThread
from PySide6.QtGui import QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QLineEdit, QFormLayout, QCheckBox, QSlider,
    QFileDialog, QTabWidget, QTextEdit, QMessageBox
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QPixmap

from pipeline import BlockDims, generate_job


class ImageRoiWidget(QWidget):
    """
    Shows an image and allows user to drag-select an ROI.
    Emits roi_changed with ROI in ORIGINAL IMAGE pixel coords: (x, y, w, h) or None.
    """
    roi_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 320)
        self._pix: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0

        self._dragging = False
        self._start = QPoint(0, 0)
        self._end = QPoint(0, 0)
        self._roi_img_xywh = None  # (x,y,w,h) in original pixels

    def set_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._pix = None
            self._roi_img_xywh = None
            self.roi_changed.emit(None)
            self.update()
            return
        self._pix = pix
        self._img_w = pix.width()
        self._img_h = pix.height()
        self._roi_img_xywh = None
        self.roi_changed.emit(None)
        self.update()

    def clear_roi(self):
        self._roi_img_xywh = None
        self.roi_changed.emit(None)
        self.update()

    def roi_img_xywh(self):
        return self._roi_img_xywh

    def _fit_rect(self):
        """
        Compute where the image is drawn (preserve aspect).
        Returns (x0, y0, scale)
        """
        if not self._pix:
            return 0, 0, 1.0
        w = self.width()
        h = self.height()
        sx = w / self._img_w
        sy = h / self._img_h
        s = min(sx, sy)
        draw_w = self._img_w * s
        draw_h = self._img_h * s
        x0 = (w - draw_w) / 2
        y0 = (h - draw_h) / 2
        return x0, y0, s

    def _widget_to_img(self, p: QPoint):
        x0, y0, s = self._fit_rect()
        x = (p.x() - x0) / s
        y = (p.y() - y0) / s
        x = max(0, min(self._img_w, x))
        y = max(0, min(self._img_h, y))
        return int(x), int(y)

    def mousePressEvent(self, e):
        if not self._pix:
            return
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._start = e.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if not self._dragging:
            return
        self._dragging = False

        p1 = self._start
        p2 = self._end
        left = min(p1.x(), p2.x())
        right = max(p1.x(), p2.x())
        top = min(p1.y(), p2.y())
        bottom = max(p1.y(), p2.y())

        # Convert to image coords
        x1, y1 = self._widget_to_img(QPoint(left, top))
        x2, y2 = self._widget_to_img(QPoint(right, bottom))

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # If selection is tiny, treat as "no ROI"
        if w < 10 or h < 10:
            self._roi_img_xywh = None
            self.roi_changed.emit(None)
        else:
            self._roi_img_xywh = (x1, y1, w, h)
            self.roi_changed.emit(self._roi_img_xywh)

        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self._pix:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        x0, y0, s = self._fit_rect()
        draw_w = self._img_w * s
        draw_h = self._img_h * s
        painter.drawPixmap(int(x0), int(y0), int(draw_w), int(draw_h), self._pix)

        # Draw ROI rect (widget coords) while dragging
        if self._dragging:
            r = QRect(self._start, self._end).normalized()
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(r)

        # Draw final ROI if set
        if self._roi_img_xywh is not None:
            x, y, w, h = self._roi_img_xywh
            # map image roi -> widget coords
            rx = x0 + x * s
            ry = y0 + y * s
            rw = w * s
            rh = h * s
            pen = QPen(Qt.yellow, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(int(rx), int(ry), int(rw), int(rh))


class DropZone(QLabel):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__("Drag & drop an image here\n(or click Browse)")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("border: 2px dashed #888; padding: 24px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            self.file_dropped.emit(path)
        else:
            QMessageBox.warning(self, "Unsupported", "Please drop a PNG/JPG/WEBP image.")


class WorkerThread(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, image_path, block, margin_mm, detail, include_top, roi_xywh):
        super().__init__()
        self.image_path = image_path
        self.block = block
        self.margin_mm = margin_mm
        self.detail = detail
        self.include_top = include_top
        self.roi_xywh = roi_xywh

    def run(self):
        try:
            res = generate_job(
                image_path=self.image_path,
                block=self.block,
                margin_mm=self.margin_mm,
                detail=self.detail,
                include_top=self.include_top,
                out_root="jobs",
                roi_xywh=self.roi_xywh
            )
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CarveStencil (GUI MVP + ROI)")
        self.image_path = None
        self.job = None
        self.worker = None
        self.current_roi = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # LEFT: controls
        left = QVBoxLayout()
        layout.addLayout(left, 1)

        self.drop = DropZone()
        self.drop.file_dropped.connect(self.on_file)
        left.addWidget(self.drop)

        browse_btn = QPushButton("Browse image…")
        browse_btn.clicked.connect(self.browse)
        left.addWidget(browse_btn)

        self.roi_status = QLabel("ROI: (none) — drag on the Input tab to select")
        left.addWidget(self.roi_status)

        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self.clear_roi)
        left.addWidget(clear_roi_btn)

        form = QFormLayout()
        left.addLayout(form)

        self.L = QLineEdit("60")
        self.W = QLineEdit("40")
        self.H = QLineEdit("40")
        form.addRow("Block L (mm)", self.L)
        form.addRow("Block W (mm)", self.W)
        form.addRow("Block H (mm)", self.H)

        self.margin = QLineEdit("3")
        form.addRow("Margin/allowance (mm)", self.margin)

        self.top_cb = QCheckBox("Include TOP template")
        self.top_cb.setChecked(True)
        left.addWidget(self.top_cb)

        left.addWidget(QLabel("Detail (simple → detailed)"))
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setMinimum(0)
        self.detail_slider.setMaximum(100)
        self.detail_slider.setValue(60)
        left.addWidget(self.detail_slider)

        self.generate_btn = QPushButton("Generate templates")
        self.generate_btn.clicked.connect(self.generate)
        self.generate_btn.setEnabled(False)
        left.addWidget(self.generate_btn)

        self.export_btn = QPushButton("Export ZIP…")
        self.export_btn.clicked.connect(self.export_zip)
        self.export_btn.setEnabled(False)
        left.addWidget(self.export_btn)

        left.addStretch(1)

        # RIGHT: preview tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 2)

        # Input tab with ROI selection
        self.input_roi = ImageRoiWidget()
        self.input_roi.roi_changed.connect(self.on_roi_changed)
        self.tabs.addTab(self.input_roi, "Input (ROI)")

        # Overlay tab (photo + polyline)
        self.overlay_label = QLabel()
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("background: #111;")
        self.tabs.addTab(self.overlay_label, "Overlay")

        self.front_view = QSvgWidget()
        self.right_view = QSvgWidget()
        self.top_view = QSvgWidget()
        self.net_view = QSvgWidget()

        self.guide_view = QTextEdit()
        self.guide_view.setReadOnly(True)

        self.tabs.addTab(self.front_view, "Front")
        self.tabs.addTab(self.right_view, "Right")
        self.tabs.addTab(self.top_view, "Top")
        self.tabs.addTab(self.net_view, "Net Layout")
        self.tabs.addTab(self.guide_view, "Guide")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.job:
            overlay_path = self.job.files.get("debug_overlay")
            if overlay_path and os.path.exists(overlay_path):
                pix = QPixmap(overlay_path)
                self.overlay_label.setPixmap(
                    pix.scaled(self.overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.on_file(path)

    def on_file(self, path):
        self.image_path = path
        self.drop.setText(f"Loaded:\n{os.path.basename(path)}")
        self.generate_btn.setEnabled(True)

        # Load image into ROI widget
        self.input_roi.set_image(path)
        self.current_roi = None
        self.roi_status.setText("ROI: (none) — drag on the Input tab to select")
        self.tabs.setCurrentIndex(0)  # go to Input tab

    def clear_roi(self):
        self.input_roi.clear_roi()

    def on_roi_changed(self, roi):
        self.current_roi = roi
        if roi is None:
            self.roi_status.setText("ROI: (none) — drag on the Input tab to select")
        else:
            x, y, w, h = roi
            self.roi_status.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")

    def generate(self):
        if not self.image_path:
            return

        try:
            block = BlockDims(L=float(self.L.text()), W=float(self.W.text()), H=float(self.H.text()))
            margin_mm = float(self.margin.text())
        except Exception:
            QMessageBox.warning(self, "Input error", "Please enter valid numeric dimensions.")
            return

        # slider mapping: 0..100 → detail 0.05..0.005
        t = self.detail_slider.value() / 100.0
        detail = 0.05 - (0.045 * t)

        self.generate_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.drop.setText("Processing…")

        self.worker = WorkerThread(
            image_path=self.image_path,
            block=block,
            margin_mm=margin_mm,
            detail=detail,
            include_top=self.top_cb.isChecked(),
            roi_xywh=self.current_roi
        )
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    def on_done(self, job):
        self.job = job
        self.drop.setText(f"Done ✓ {job.job_id}")

        self.front_view.load(job.files["front_svg"])
        self.right_view.load(job.files["right_svg"])
        if job.files["top_svg"]:
            self.top_view.load(job.files["top_svg"])
        else:
            self.top_view.load("")

        self.net_view.load(job.files["net_svg"])

        with open(job.files["guide_md"], "r", encoding="utf-8") as f:
            self.guide_view.setMarkdown(f.read())

        # Load overlay preview (ROI image with polyline)
        overlay_path = job.files.get("debug_overlay")
        if overlay_path and os.path.exists(overlay_path):
            pix = QPixmap(overlay_path)
            self.overlay_label.setPixmap(
                pix.scaled(self.overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.overlay_label.setText("")  # clear any placeholder text
        else:
            self.overlay_label.setPixmap(QPixmap())
            self.overlay_label.setText("No overlay generated.\n(Check pipeline.py adds debug_overlay)")

        self.export_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.tabs.setCurrentIndex(1)  # go to Front

    def on_fail(self, msg):
        self.drop.setText("Failed ✗")
        QMessageBox.critical(self, "Processing failed", msg)
        self.generate_btn.setEnabled(True)

    def export_zip(self):
        if not self.job:
            return
        default_name = os.path.basename(self.job.files["zip"])
        out_path, _ = QFileDialog.getSaveFileName(self, "Save ZIP", default_name, "ZIP (*.zip)")
        if not out_path:
            return
        shutil.copyfile(self.job.files["zip"], out_path)
        QMessageBox.information(self, "Saved", f"Exported:\n{out_path}")


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    app.exec()
