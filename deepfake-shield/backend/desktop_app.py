import sys
import os
import io
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QTabWidget,
    QProgressBar, QFrame, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal

API_BASE = "http://127.0.0.1:8000/api/v1"

# --- Async Worker Thread to Prevent GUI Freezing ---
class ForensicWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, endpoint: str, file_path: str = None, text_data: str = None):
        super().__init__()
        self.endpoint = endpoint
        self.file_path = file_path
        self.text_data = text_data

    def run(self):
        try:
            if self.file_path:
                with open(self.file_path, "rb") as f:
                    files = {"file": (os.path.basename(self.file_path), f.read())}
                    resp = requests.post(f"{API_BASE}/{self.endpoint}", files=files, timeout=40)
            elif self.text_data:
                data = {"text": self.text_data}
                resp = requests.post(f"{API_BASE}/{self.endpoint}", data=data, timeout=15)
            else:
                self.error.emit("No input payload provided.")
                return

            if resp.status_code == 200:
                self.finished.emit(resp.json())
            else:
                self.error.emit(f"Server Error HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            self.error.emit(f"Connection Failed: Ensure backend is running. ({str(e)})")


# --- Main Software Application Window ---
class DeepfakeShieldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake & Misinformation Shield - Forensic Suite")
        self.resize(1000, 720)
        self.selected_image_path = None
        self.selected_audio_path = None
        self.selected_video_path = None

        self.apply_dark_theme()
        self.init_ui()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; color: #f1f5f9; }
            QTabWidget::pane { border: 1px solid #1e293b; background: #0f172a; border-radius: 8px; }
            QTabBar::tab {
                background: #1e293b; color: #94a3b8; padding: 10px 24px; font-weight: bold;
                border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px;
            }
            QTabBar::tab:selected { background: #0284c7; color: #ffffff; }
            QPushButton {
                background-color: #0284c7; color: white; font-weight: bold; padding: 10px 18px;
                border-radius: 6px; border: none; font-size: 13px;
            }
            QPushButton:hover { background-color: #0369a1; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
            QTextEdit {
                background-color: #1e293b; border: 1px solid #334155; border-radius: 6px;
                padding: 10px; color: #f8fafc; font-size: 13px;
            }
            QProgressBar {
                border: 1px solid #334155; border-radius: 4px; text-align: center; background: #1e293b;
            }
            QProgressBar::chunk { background-color: #0284c7; }
        """)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Header Title Banner
        header = QLabel("🛡️ Deepfake & Misinformation Shield — Multimodal Verifier")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #38bdf8; margin-bottom: 8px;")
        main_layout.addWidget(header)

        # Tab Setup
        tabs = QTabWidget()
        tabs.addTab(self.create_image_tab(), "🖼️ Image Forensics")
        tabs.addTab(self.create_audio_tab(), "🎙️ Audio Clones")
        tabs.addTab(self.create_video_tab(), "🎥 Video Analysis")
        tabs.addTab(self.create_claim_tab(), "📰 Claim Fact-Check")
        main_layout.addWidget(tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # 1. IMAGE FORENSIC TAB
    def create_image_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()

        # Left Column: Upload & Canvas Preview
        left_box = QVBoxLayout()
        self.img_preview = QLabel("No Image Selected")
        self.img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setStyleSheet("border: 2px dashed #334155; border-radius: 8px; background: #111827;")
        self.img_preview.setFixedSize(480, 480)
        left_box.addWidget(self.img_preview)

        btn_select = QPushButton("Choose Image File")
        btn_select.clicked.connect(self.select_image)
        left_box.addWidget(btn_select)

        self.btn_run_img = QPushButton("Scan Image for Deepfakes")
        self.btn_run_img.setEnabled(False)
        self.btn_run_img.clicked.connect(self.run_image_scan)
        left_box.addWidget(self.btn_run_img)

        layout.addLayout(left_box)

        # Right Column: Metric Breakdown
        self.img_result_pane = self.create_result_pane()
        layout.addWidget(self.img_result_pane)

        tab.setLayout(layout)
        return tab

    # 2. AUDIO FORENSIC TAB
    def create_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.lbl_audio = QLabel("Select an audio track (.wav, .mp3, .ogg) to detect neural voice cloning:")
        layout.addWidget(self.lbl_audio)

        btn_select_audio = QPushButton("Select Audio File")
        btn_select_audio.clicked.connect(self.select_audio)
        layout.addWidget(btn_select_audio)

        self.btn_run_audio = QPushButton("Inspect Vocal Frequencies")
        self.btn_run_audio.setEnabled(False)
        self.btn_run_audio.clicked.connect(self.run_audio_scan)
        layout.addWidget(self.btn_run_audio)

        self.audio_result_pane = self.create_result_pane()
        layout.addWidget(self.audio_result_pane)

        tab.setLayout(layout)
        return tab

    # 3. VIDEO FORENSIC TAB
    def create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.lbl_video = QLabel("Upload an MP4 video to evaluate temporal frame-by-frame manipulation:")
        layout.addWidget(self.lbl_video)

        btn_select_video = QPushButton("Select Video File")
        btn_select_video.clicked.connect(self.select_video)
        layout.addWidget(btn_select_video)

        self.btn_run_video = QPushButton("Run Temporal Video Inspection")
        self.btn_run_video.setEnabled(False)
        self.btn_run_video.clicked.connect(self.run_video_scan)
        layout.addWidget(self.btn_run_video)

        self.video_result_pane = self.create_result_pane()
        layout.addWidget(self.video_result_pane)

        tab.setLayout(layout)
        return tab

    # 4. CLAIM GROUNDING TAB
    def create_claim_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Paste suspicious news copy, forwarded text, or viral claim:"))
        self.txt_claim = QTextEdit()
        self.txt_claim.setPlaceholderText("Enter claim statement here...")
        self.txt_claim.setFixedHeight(120)
        layout.addWidget(self.txt_claim)

        btn_run_claim = QPushButton("Verify Against Fact-Checking Databases")
        btn_run_claim.clicked.connect(self.run_claim_scan)
        layout.addWidget(btn_run_claim)

        self.claim_result_pane = self.create_result_pane()
        layout.addWidget(self.claim_result_pane)

        tab.setLayout(layout)
        return tab

    def create_result_pane(self):
        frame = QFrame()
        frame.setStyleSheet("background-color: #111827; border-radius: 8px; padding: 12px; border: 1px solid #1e293b;")
        vbox = QVBoxLayout()

        badge = QLabel("STATUS: IDLE")
        badge.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        badge.setStyleSheet("color: #94a3b8;")
        vbox.addWidget(badge)

        exp = QLabel("Run a scan to see plain-language evidence and confidence scores.")
        exp.setWordWrap(True)
        exp.setStyleSheet("color: #cbd5e1; font-size: 13px; margin-top: 6px;")
        vbox.addWidget(exp)

        details = QTextEdit()
        details.setReadOnly(True)
        details.setStyleSheet("background-color: #0b0f19; border: 1px solid #1e293b; color: #38bdf8; font-family: Consolas;")
        vbox.addWidget(details)

        frame.badge = badge
        frame.exp = exp
        frame.details = details
        frame.setLayout(vbox)
        return frame

    # --- UI Event Handlers ---
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.selected_image_path = path
            pixmap = QPixmap(path).scaled(480, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_preview.setPixmap(pixmap)
            self.btn_run_img.setEnabled(True)

    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.wav *.mp3 *.ogg)")
        if path:
            self.selected_audio_path = path
            self.lbl_audio.setText(f"Selected: {os.path.basename(path)}")
            self.btn_run_audio.setEnabled(True)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.mkv *.webm)")
        if path:
            self.selected_video_path = path
            self.lbl_video.setText(f"Selected: {os.path.basename(path)}")
            self.btn_run_video.setEnabled(True)

    def run_image_scan(self):
        self.set_loading(self.img_result_pane, "Analyzing image pixels & artifacts...")
        self.worker = ForensicWorker(endpoint="detect/image", file_path=self.selected_image_path)
        self.worker.finished.connect(self.render_image_result)
        self.worker.error.connect(lambda err: self.render_error(self.img_result_pane, err))
        self.worker.start()

    def run_audio_scan(self):
        self.set_loading(self.audio_result_pane, "Evaluating vocoder phase & harmonics...")
        self.worker = ForensicWorker(endpoint="detect/audio", file_path=self.selected_audio_path)
        self.worker.finished.connect(lambda res: self.render_generic_result(self.audio_result_pane, res))
        self.worker.error.connect(lambda err: self.render_error(self.audio_result_pane, err))
        self.worker.start()

    def run_video_scan(self):
        self.set_loading(self.video_result_pane, "Sampling frames & detecting facial consistency...")
        self.worker = ForensicWorker(endpoint="detect/video", file_path=self.selected_video_path)
        self.worker.finished.connect(lambda res: self.render_generic_result(self.video_result_pane, res))
        self.worker.error.connect(lambda err: self.render_error(self.video_result_pane, err))
        self.worker.start()

    def run_claim_scan(self):
        text = self.txt_claim.toPlainText().strip()
        if not text:
            return
        self.set_loading(self.claim_result_pane, "Querying global fact-check registries...")
        self.worker = ForensicWorker(endpoint="detect/text", text_data=text)
        self.worker.finished.connect(lambda res: self.render_claim_result(self.claim_result_pane, res))
        self.worker.error.connect(lambda err: self.render_error(self.claim_result_pane, err))
        self.worker.start()

    # --- Renderers ---
    def set_loading(self, pane, msg):
        pane.badge.setText("STATUS: SCANNING IN PROGRESS...")
        pane.badge.setStyleSheet("color: #f59e0b;")
        pane.exp.setText(msg)
        pane.details.clear()

    def render_error(self, pane, err):
        pane.badge.setText("STATUS: FAILED")
        pane.badge.setStyleSheet("color: #ef4444;")
        pane.exp.setText(err)

    def render_image_result(self, data):
        verdict = data.get("verdict", "UNKNOWN")
        conf = data.get("confidence", 0.0)
        is_bad = verdict in ["MANIPULATED", "SYNTHETIC"]

        self.img_result_pane.badge.setText(f"VERDICT: {verdict} ({conf*100:.1f}%)")
        self.img_result_pane.badge.setStyleSheet(f"color: {'#ef4444' if is_bad else '#10b981'};")
        self.img_result_pane.exp.setText(data.get("plain_explanation", ""))
        self.img_result_pane.details.setPlainText(str(data))

        # Render Bounding Box Over Image on Desktop Canvas
        regions = data.get("suspicious_regions", [])
        if self.selected_image_path and os.path.exists(self.selected_image_path):
            pixmap = QPixmap(self.selected_image_path).scaled(480, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            if regions:
                orig_img = QImage(self.selected_image_path)
                scale_x = pixmap.width() / orig_img.width()
                scale_y = pixmap.height() / orig_img.height()

                painter = QPainter(pixmap)
                pen = QPen(QColor(239, 68, 68), 3)
                painter.setPen(pen)
                painter.setBrush(QColor(239, 68, 68, 60))

                for r in regions:
                    y1, x1, y2, x2 = r["box_2d"]
                    bx = int(x1 * scale_x)
                    by = int(y1 * scale_y)
                    bw = int((x2 - x1) * scale_x)
                    bh = int((y2 - y1) * scale_y)
                    painter.drawRect(bx, by, bw, bh)

                painter.end()

            self.img_preview.setPixmap(pixmap)

    def render_generic_result(self, pane, data):
        verdict = data.get("verdict", "UNKNOWN")
        conf = data.get("confidence", 0.0)
        is_bad = verdict in ["MANIPULATED", "SYNTHETIC"]

        pane.badge.setText(f"VERDICT: {verdict} ({conf*100:.1f}%)")
        pane.badge.setStyleSheet(f"color: {'#ef4444' if is_bad else '#10b981'};")
        pane.exp.setText(data.get("plain_explanation", ""))
        pane.details.setPlainText(str(data))

    def render_claim_result(self, pane, data):
        verdict = data.get("verdict", "UNKNOWN")
        is_bad = verdict == "DEBUNKED"

        pane.badge.setText(f"VERDICT: {verdict}")
        pane.badge.setStyleSheet(f"color: {'#ef4444' if is_bad else '#10b981'};")
        pane.exp.setText(data.get("plain_explanation", ""))

        matches = data.get("fact_check", {}).get("matches", [])
        details_txt = ""
        for m in matches:
            details_txt += f"• Source: {m.get('publisher')}\n  Rating: {m.get('rating')}\n  URL: {m.get('url')}\n\n"

        pane.details.setPlainText(details_txt if details_txt else "No international debunk records matched.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeShieldApp()
    window.show()
    sys.exit(app.exec())