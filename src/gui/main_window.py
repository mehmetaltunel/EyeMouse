"""
Eye Mouse - Professional GUI Application
PyQt6-based cross-platform interface
"""

import sys
import cv2
import numpy as np
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSlider, QGroupBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QKeySequence, QShortcut
from pynput import keyboard

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import AppConfig, config
from src.core.eye_tracker import EyeTracker, GazeData
from src.core.blink_detector import BlinkDetector, BlinkType
from src.core.mouse_controller import MouseController
from src.core.calibration import Calibrator, GazeQuality
from src.utils.camera_utils import get_available_cameras


class CameraWorker(QThread):
    """Background camera processing thread"""
    frame_ready = pyqtSignal(np.ndarray, bool, object, tuple)  # frame, face_found, gaze_data, ear
    blink_detected = pyqtSignal(object)  # BlinkType
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.running = False
        self.eye_tracker: Optional[EyeTracker] = None
        self.blink_detector: Optional[BlinkDetector] = None
        self.cap = None
        
    def run(self):
        # Initialize components
        self.eye_tracker = EyeTracker()
        self.blink_detector = BlinkDetector(
            ear_threshold=self.config.tracking.ear_threshold,
            consecutive_frames=self.config.tracking.ear_consecutive_frames,
            cooldown=self.config.tracking.blink_cooldown
        )
        
        self.cap = cv2.VideoCapture(self.config.camera.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Process
            face_found, gaze_data = self.eye_tracker.process(frame)
            
            ear = (0.3, 0.3)
            
            if face_found and gaze_data:
                # Draw overlay
                if self.config.show_landmarks:
                    frame = self.eye_tracker.draw_overlay(frame, True)
                    
                # Blink detection
                left_lm = self.eye_tracker.get_eye_landmarks('left')
                right_lm = self.eye_tracker.get_eye_landmarks('right')
                
                if left_lm and right_lm:
                    blink = self.blink_detector.detect(left_lm, right_lm)
                    if blink:
                        self.blink_detected.emit(blink.blink_type)
                        
                ear = self.blink_detector.get_ear_values()
                
            self.frame_ready.emit(frame, face_found, gaze_data, ear)
            
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.eye_tracker:
            self.eye_tracker.close()
            
    def stop(self):
        self.running = False
        self.wait()
        
    def update_thresholds(self, ear_threshold: float, cooldown: float):
        if self.blink_detector:
            self.blink_detector.update_thresholds(ear_threshold, cooldown)

    def update_sensitivity(self, value: float):
        if self.eye_tracker:
            self.eye_tracker.set_sensitivity(value)

class HotkeysWorker(QThread):
    """Global keyboard shortcuts listener"""
    toggle_triggered = pyqtSignal()
    calibrate_triggered = pyqtSignal()
    
    def run(self):
        self.pressed_keys = set()
        
        # Start listener
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        ) as listener:
            self.listener = listener
            listener.join()
            
    def _on_press(self, key):
        self.pressed_keys.add(key)
        
        # Check for shortcuts
        # Cmd/Ctrl + Shift + X -> Toggle
        # Cmd/Ctrl + Shift + C -> Calibrate
        
        is_cmd = keyboard.Key.cmd in self.pressed_keys or keyboard.Key.cmd_l in self.pressed_keys or keyboard.Key.cmd_r in self.pressed_keys
        is_ctrl = keyboard.Key.ctrl in self.pressed_keys or keyboard.Key.ctrl_l in self.pressed_keys or keyboard.Key.ctrl_r in self.pressed_keys
        is_mod = is_cmd or is_ctrl
        
        is_shift = keyboard.Key.shift in self.pressed_keys or keyboard.Key.shift_l in self.pressed_keys or keyboard.Key.shift_r in self.pressed_keys
        
        if is_mod and is_shift:
            if hasattr(key, 'char'):
                if key.char == 'x' or key.char == 'X':
                    self.toggle_triggered.emit()
                elif key.char == 'c' or key.char == 'C':
                    self.calibrate_triggered.emit()
                    
    def _on_release(self, key):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        
    def stop(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.wait()


class CalibrationOverlay(QWidget):
    """Full-screen calibration overlay with gaze visualization"""
    calibration_complete = pyqtSignal()
    
    def __init__(self, calibrator: Calibrator, screen_geometry):
        super().__init__()
        self.calibrator = calibrator
        self.screen_geometry = screen_geometry
        
        self.current_point = None
        self.progress = 0.0
        self.gaze_quality = GazeQuality.NO_GAZE
        self.current_gaze = None
        self.smooth_gaze = None  # Smoothed gaze for display
        
        # macOS true fullscreen - covers dock and menubar
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.BypassWindowManagerHint  # This covers dock on macOS
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setGeometry(screen_geometry)
        self.move(screen_geometry.x(), screen_geometry.y())
        self.setStyleSheet("background-color: #0a0a12;")
        
        # Info panel at bottom
        self.info_panel = QWidget(self)
        self.info_panel.setStyleSheet("background-color: rgba(0,0,0,0.7); border-radius: 8px;")
        
        # Instruction
        self.instruction = QLabel("Look at the target circle", self.info_panel)
        self.instruction.setStyleSheet("""
            color: #fff;
            font-size: 16px;
            font-weight: 500;
            background: transparent;
        """)
        self.instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Progress/status
        self.progress_label = QLabel("", self.info_panel)
        self.progress_label.setStyleSheet("color: #888; font-size: 13px; background: transparent;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel (ESC)", self.info_panel)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #888;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #444;
                color: #fff;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel)
        
        self._layout_widgets()
        
    def _layout_widgets(self):
        w = self.width()
        h = self.height()
        
        # Info panel at bottom center
        panel_w = 300
        panel_h = 90
        panel_x = (w - panel_w) // 2
        panel_y = h - panel_h - 30
        self.info_panel.setGeometry(panel_x, panel_y, panel_w, panel_h)
        
        # Layout inside panel
        self.instruction.setGeometry(10, 10, panel_w - 20, 25)
        self.progress_label.setGeometry(10, 35, panel_w - 20, 20)
        self.cancel_btn.setGeometry(panel_w // 2 - 50, 58, 100, 26)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_widgets()
        
    def showEvent(self, event):
        super().showEvent(event)
        # Force fullscreen on macOS
        self.setGeometry(self.screen_geometry)
        self.setFixedSize(self.screen_geometry.width(), self.screen_geometry.height())
        self.move(0, 0)
        self.raise_()
        self.activateWindow()
        self.showFullScreen()
        self._layout_widgets()
        
    def update_calibration(self, gaze_data):
        self.current_gaze = gaze_data
        if gaze_data:
            gaze_pos = gaze_data.position
        else:
            gaze_pos = None
            
        done, point, progress = self.calibrator.update(gaze_pos)
        
        if done:
            self.calibration_complete.emit()
            self.close()
            return
            
        self.current_point = point
        self.progress = progress
        
        # Get gaze quality
        quality = self.calibrator.get_gaze_quality()
        self.gaze_quality = quality
        
        # Update status text
        current, total = self.calibrator.get_progress_info()
        
        # Format info text
        if self.current_gaze:
            gx, gy = self.current_gaze.position
            dist_str = ""
            if hasattr(self.current_gaze, 'distance'):
                 d = self.current_gaze.distance
                 dist_str = f" | Dist: {d:.2f}"
                 if d < 0.8: dist_str += " (Close)"
                 elif d > 1.2: dist_str += " (Far)"
            
            coord_str = f"Gaze: ({gx:.2f}, {gy:.2f}){dist_str}"
        else:
            coord_str = "No gaze detected"
        
        if quality == GazeQuality.NO_GAZE:
            self.progress_label.setText(f"[{current}/{total}] Face not detected\n{coord_str}")
            self.progress_label.setStyleSheet("color: #e63946; font-size: 13px; background: transparent;")
        elif quality == GazeQuality.OFF_TARGET:
            self.progress_label.setText(f"[{current}/{total}] Look at the circle!\n{coord_str}")
            self.progress_label.setStyleSheet("color: #fca311; font-size: 13px; background: transparent;")
        elif quality == GazeQuality.NEAR_TARGET:
            self.progress_label.setText(f"[{current}/{total}] Getting closer...\n{coord_str}")
            self.progress_label.setStyleSheet("color: #00b4d8; font-size: 13px; background: transparent;")
        else:  # ON_TARGET
            pct = int(progress * 100)
            self.progress_label.setText(f"[{current}/{total}] Hold steady! {pct}%\n{coord_str}")
            self.progress_label.setStyleSheet("color: #00d26a; font-size: 13px; font-weight: bold; background: transparent;")
        
        self.update()
        
    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QBrush, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw gaze cursor (red dot showing where you're looking)
        if self.current_gaze is not None:
            # Extract position from GazeData object
            if hasattr(self.current_gaze, 'position'):
                gx, gy = self.current_gaze.position
            else:
                gx, gy = self.current_gaze
                
            gaze_x = int(gx * w)
            gaze_y = int(gy * h)
            
            # Outer glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 50, 50, 100)))
            painter.drawEllipse(gaze_x - 20, gaze_y - 20, 40, 40)
            
            # Inner dot
            painter.setBrush(QBrush(QColor(255, 50, 50)))
            painter.drawEllipse(gaze_x - 8, gaze_y - 8, 16, 16)
            
            # Center
            painter.setBrush(QBrush(QColor(255, 200, 200)))
            painter.drawEllipse(gaze_x - 3, gaze_y - 3, 6, 6)
        
        # Draw target point
        if self.current_point:
            x, y = self.current_point
            
            # Outer ring
            ring_color = QColor(60, 60, 80)
            if self.gaze_quality == GazeQuality.ON_TARGET:
                ring_color = QColor(0, 180, 100)
            elif self.gaze_quality == GazeQuality.NEAR_TARGET:
                ring_color = QColor(0, 150, 200)
                
            painter.setPen(QPen(ring_color, 3))
            painter.setBrush(QBrush(QColor(20, 20, 30)))
            painter.drawEllipse(x - 35, y - 35, 70, 70)
            
            # Progress arc
            painter.setPen(QPen(QColor(0, 220, 100), 5))
            span_angle = int(self.progress * 360 * 16)
            painter.drawArc(x - 35, y - 35, 70, 70, 90 * 16, -span_angle)
            
            # Center target dot
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 220, 255)))
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            
            # Crosshair
            painter.setPen(QPen(QColor(0, 220, 255, 150), 1))
            painter.drawLine(x - 20, y, x - 10, y)
            painter.drawLine(x + 10, y, x + 20, y)
            painter.drawLine(x, y - 20, x, y - 10)
            painter.drawLine(x, y + 10, x, y + 20)
            
    def cancel(self):
        self.calibrator.stop()
        self.close()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize config
        self.config = AppConfig()
        self.config.load()
        
        # Initialize components
        self.mouse_controller = MouseController(
            sensitivity=self.config.mouse.sensitivity,
            smoothing_samples=self.config.mouse.smoothing_samples,
            dead_zone=self.config.mouse.dead_zone,
            acceleration_curve=self.config.mouse.acceleration_curve
        )
        
        screen_w, screen_h = self.mouse_controller.get_screen_size()
        self.calibrator = Calibrator(
            screen_width=screen_w,
            screen_height=screen_h,
            points_count=self.config.calibration.points_count,
            hold_duration=self.config.calibration.point_duration
        )
        self.calibrator.load("calibration.json")
        
        # State
        self.current_gaze = None
        self.calibration_overlay: Optional[CalibrationOverlay] = None
        
        # Setup UI
        self._setup_ui()
        self._setup_shortcuts()
        
        # Start camera
        self.camera_worker = CameraWorker(self.config)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.blink_detected.connect(self._on_blink)
        self.camera_worker.start()
        
        # Start hotkeys
        self.hotkeys_worker = HotkeysWorker()
        self.hotkeys_worker.toggle_triggered.connect(self._toggle_mouse)
        self.hotkeys_worker.calibrate_triggered.connect(self._start_calibration)
        self.hotkeys_worker.start()
        
    def _setup_ui(self):
        self.setWindowTitle("Eye Mouse")
        self.setFixedSize(500, 700)  # Increased size further for comfort
        
        # Dark theme
        self.setStyleSheet(self._get_stylesheet())
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header
        header = QLabel("Eye Mouse")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        tagline = QLabel("Gaze-controlled cursor")
        tagline.setObjectName("tagline")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tagline)
        
        layout.addSpacing(8)
        
        # Camera selection
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("secondaryBtn") # Use same style
        self._load_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_change)
        cam_layout.addWidget(self.camera_combo, 1)
        
        layout.addLayout(cam_layout)
        
        # Camera preview
        self.camera_view = QLabel()
        self.camera_view.setObjectName("cameraView")
        self.camera_view.setFixedSize(348, 196)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setText("Initializing camera...")
        layout.addWidget(self.camera_view, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Status panel
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(6)
        
        # Face detection
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Face Detection"))
        self.face_indicator = QLabel("--")
        self.face_indicator.setObjectName("indicator")
        row1.addWidget(self.face_indicator)
        row1.addStretch()
        status_layout.addLayout(row1)
        
        # EAR values
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Eye Openness"))
        self.ear_label = QLabel("L: --  R: --")
        self.ear_label.setObjectName("indicator")
        row2.addWidget(self.ear_label)
        row2.addStretch()
        status_layout.addLayout(row2)
        
        # Last action
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Last Action"))
        self.action_label = QLabel("--")
        self.action_label.setObjectName("action")
        row3.addWidget(self.action_label)
        row3.addStretch()
        status_layout.addLayout(row3)
        
        # Calibration status
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Calibration"))
        self.cal_label = QLabel("Not calibrated")
        self.cal_label.setObjectName("indicator")
        row4.addWidget(self.cal_label)
        row4.addStretch()
        status_layout.addLayout(row4)
        
        # Tracking Info (Coords + Distance)
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Tracking"))
        self.track_label = QLabel("Pos: --  Dist: --")
        self.track_label.setObjectName("indicator")
        row5.addWidget(self.track_label)
        row5.addStretch()
        status_layout.addLayout(row5)
        
        layout.addWidget(status_group)
        
        # Sensitivity slider
        sens_group = QGroupBox("Sensitivity")
        sens_layout = QVBoxLayout(sens_group)
        
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setMinimum(10)
        self.sens_slider.setMaximum(40)
        self.sens_slider.setValue(int(self.config.mouse.sensitivity * 10))
        self.sens_slider.valueChanged.connect(self._on_sensitivity_change)
        sens_layout.addWidget(self.sens_slider)
        
        self.sens_label = QLabel(f"{self.config.mouse.sensitivity:.1f}x")
        self.sens_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sens_label.setObjectName("sensValue")
        sens_layout.addWidget(self.sens_label)
        
        layout.addWidget(sens_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Calibrate (C)")
        self.calibrate_btn.setObjectName("secondaryBtn")
        self.calibrate_btn.clicked.connect(self._start_calibration)
        btn_layout.addWidget(self.calibrate_btn)
        
        self.toggle_btn = QPushButton("Start (M)")
        self.toggle_btn.setObjectName("primaryBtn")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._toggle_mouse)
        btn_layout.addWidget(self.toggle_btn)
        
        layout.addLayout(btn_layout)
        
        # Footer info
        footer = QLabel("Left blink = Left click  |  Right blink = Right click")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)
        
        # Update calibration status
        self._update_calibration_status()
        
    def _get_stylesheet(self) -> str:
        return """
            QMainWindow {
                background-color: #12121a;
            }
            QWidget {
                color: #e0e0e0;
                font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 14px;  /* Increased from 13px */
            }
            #header {
                font-size: 26px; /* Increased from 24px */
                font-weight: 600;
                color: #fff;
                letter-spacing: -0.5px;
            }
            #tagline {
                font-size: 14px;
                color: #888;
            }
            #cameraView {
                background-color: #0a0a10;
                border: 1px solid #2a2a3a;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: 500;
                font-size: 15px; /* Increased header size */
                border: 1px solid #2a2a3a;
                border-radius: 8px;
                margin-top: 24px; /* More spacing */
                padding-top: 14px;
                background-color: #16161e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #888;
            }
            QComboBox {
                background-color: #22222e;
                border: 1px solid #3a3a4a;
                border-radius: 6px;
                padding: 6px 12px;
                color: #fff;
                font-size: 14px;
            }
            #indicator {
                color: #00b4d8;
                font-weight: 500;
                font-size: 14px;
            }
            #primaryBtn {
                background-color: #00b4d8;
                color: #000;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-weight: 600;
                font-size: 15px;
            }
            #secondaryBtn {
                background-color: transparent;
                color: #aaa;
                border: 1px solid #3a3a4a;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
                font-size: 14px;
            }
            #footer {
                color: #666;
                font-size: 12px;
            }
        """
        
    def _setup_shortcuts(self):
        # Q - Quit
        QShortcut(QKeySequence("Q"), self).activated.connect(self.close)
        
        # C - Calibrate
        QShortcut(QKeySequence("C"), self).activated.connect(self._start_calibration)
        
        # M - Toggle mouse
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_mouse)
        
        # D - Debug mode
        QShortcut(QKeySequence("D"), self).activated.connect(self._toggle_debug)
        
        # Escape - Stop
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._stop_mouse)
        
    def _load_cameras(self):
        """Populate camera combo box"""
        try:
            cameras = get_available_cameras()
            
            self.camera_combo.blockSignals(True)
            self.camera_combo.clear()
            
            current_idx = self.config.camera.index
            found = False
            
            for idx, name in cameras:
                self.camera_combo.addItem(f"Camera {idx}", idx)
                if idx == current_idx:
                    self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
                    found = True
                    
            if not found and self.camera_combo.count() > 0:
                self.camera_combo.setCurrentIndex(0)
                # Update config to match reality
                self.config.camera.index = self.camera_combo.currentData()
                
            self.camera_combo.blockSignals(False)
        except Exception as e:
            print(f"Error loading cameras: {e}")
            self.camera_combo.addItem("Default Camera", 0)

    def _on_camera_change(self, index):
        """Handle camera selection change"""
        if index < 0: return
        
        cam_idx = self.camera_combo.currentData()
        
        # Update config
        self.config.camera.index = cam_idx
        self.config.save()
        
        # Restart camera worker
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
            
        # Re-initialize
        self.camera_worker = CameraWorker(self.config)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.blink_detected.connect(self._on_blink)
        self.camera_worker.start()

    def _on_frame(self, frame: np.ndarray, face_found: bool, gaze_data, ear: tuple):
        # Update camera view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = img.scaled(348, 196, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self.camera_view.setPixmap(QPixmap.fromImage(scaled))
        
        # Update status labels
        self.face_indicator.setText("Detected" if face_found else "Searching...")
        self.face_indicator.setStyleSheet(
            f"color: {'#00d26a' if face_found else '#e63946'}; font-weight: 500;"
        )
        
        if gaze_data:
            gx, gy = gaze_data.position
            dist_str = "OK"
            d = 1.0
            if hasattr(gaze_data, 'distance'):
                d = gaze_data.distance
                if d < 0.8: dist_str = "Close"
                elif d > 1.2: dist_str = "Far"
            
            self.track_label.setText(f"Pos: ({gx:.2f}, {gy:.2f})  Dist: {d:.2f} ({dist_str})")
        else:
             self.track_label.setText("Pos: --  Dist: --")
            
        if ear:
            self.ear_label.setText(f"L: {ear[0]:.2f}  R: {ear[1]:.2f}")
        
        # Handle gaze
        if gaze_data and face_found:
            self.current_gaze = gaze_data.position
            
            # Calibration mode
            if self.calibration_overlay and self.calibration_overlay.isVisible():
                self.calibration_overlay.update_calibration(gaze_data)
                
            # Mouse control
            elif self.mouse_controller.is_enabled():
                gx, gy = gaze_data.position
                self.mouse_controller.move_to_gaze(gx, gy)
                
    def _on_blink(self, blink_type):
        if blink_type == BlinkType.LEFT:
            self.action_label.setText("Left Click")
            self.action_label.setStyleSheet("color: #00d26a;")
            if self.mouse_controller.is_enabled():
                self.mouse_controller.click('left')
        elif blink_type == BlinkType.RIGHT:
            self.action_label.setText("Right Click")
            self.action_label.setStyleSheet("color: #fca311;")
            if self.mouse_controller.is_enabled():
                self.mouse_controller.click('right')
        elif blink_type == BlinkType.BOTH:
            self.action_label.setText("Double Click")
            self.action_label.setStyleSheet("color: #e63946;")
            if self.mouse_controller.is_enabled():
                self.mouse_controller.double_click()
                
        # Clear after delay
        QTimer.singleShot(1500, lambda: self.action_label.setText("--"))
        
    def _toggle_mouse(self):
        enabled = not self.mouse_controller.is_enabled()
        self.mouse_controller.set_enabled(enabled)
        
        if enabled:
            self.toggle_btn.setText("Stop (M)")
            self.toggle_btn.setChecked(True)
        else:
            self.toggle_btn.setText("Start (M)")
            self.toggle_btn.setChecked(False)
            
    def _stop_mouse(self):
        self.mouse_controller.set_enabled(False)
        self.toggle_btn.setText("Start (M)")
        self.toggle_btn.setChecked(False)
        
    def _toggle_debug(self):
        self.config.show_landmarks = not self.config.show_landmarks
        
    def _on_sensitivity_change(self, value: int):
        # Value is 1-10
        # Update config
        self.config.mouse.sensitivity = float(value)
        self.config.save()
        
        # Update mouse controller (acceleration curve)
        self.mouse_controller.update_params(sensitivity=float(value))
        
        # Update eye tracker (active area) - THIS IS THE BOOST
        if hasattr(self, 'camera_worker') and self.camera_worker.isRunning():
            self.camera_worker.update_sensitivity(float(value))
            
        # Update label
        self.sens_value.setText(f"{float(value):.1f}x")
        
    def _start_calibration(self):
        # Stop mouse control during calibration
        self._stop_mouse()
        
        # Start calibration
        self.calibrator.start()
        
        # Get primary screen geometry
        screen = QApplication.primaryScreen()
        screen_geo = screen.geometry()
        
        # Show overlay
        self.calibration_overlay = CalibrationOverlay(self.calibrator, screen_geo)
        self.calibration_overlay.calibration_complete.connect(self._on_calibration_complete)
        self.calibration_overlay.show()
        
    def _on_calibration_complete(self):
        self.calibrator.save("calibration.json")
        self._update_calibration_status()
        
    def _update_calibration_status(self):
        if self.calibrator.is_calibrated():
            self.cal_label.setText("Ready")
            self.cal_label.setStyleSheet("color: #00d26a; font-weight: 500;")
        else:
            self.cal_label.setText("Not calibrated")
            self.cal_label.setStyleSheet("color: #888; font-weight: 500;")
            
    def closeEvent(self, event):
        self.camera_worker.stop()
        if hasattr(self, 'hotkeys_worker'):
            self.hotkeys_worker.stop()
        self.config.save()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(18, 18, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(10, 10, 16))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(34, 34, 46))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
