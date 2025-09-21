# custom/receiver.py
"""
KITTI 3D Object Detection Receiver - Real-time Display with Controls
Receives streaming data via ZeroMQ and displays with navigation controls

Dependencies:
pip install zmq opencv-python numpy PyQt5

Usage:
python kitti_receiver.py --port 5555
"""

import sys
import argparse
import json
import time
import threading
import queue
from collections import deque
import zmq
import cv2
import numpy as np
import os

# GUI imports
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                                 QWidget, QLabel, QPushButton, QSlider, QSpinBox,
                                 QStatusBar, QFrame, QSplitter, QTextEdit, QCheckBox,
                                 QGroupBox, QGridLayout, QLCDNumber, QProgressBar,
                                 QFileDialog, QMessageBox)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
except ImportError:
    print("ERROR: PyQt5 not installed. Please install with:")
    print("pip install PyQt5")
    sys.exit(1)


class FrameBuffer:
    """Thread-safe frame buffer with history."""

    def __init__(self, max_size=100):
        self.frames = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.current_index = -1

    def add_frame(self, frame_data):
        """Add new frame to buffer."""
        with self.lock:
            self.frames.append(frame_data)
            self.current_index = len(self.frames) - 1

    def get_current_frame(self):
        """Get current frame."""
        with self.lock:
            if self.current_index >= 0 and self.current_index < len(self.frames):
                return self.frames[self.current_index]
        return None

    def get_frame_at(self, index):
        """Get frame at specific index."""
        with self.lock:
            if 0 <= index < len(self.frames):
                return self.frames[index]
        return None

    def set_current_index(self, index):
        """Set current frame index."""
        with self.lock:
            if 0 <= index < len(self.frames):
                self.current_index = index
                return True
        return False

    def get_frame_count(self):
        """Get total frame count."""
        with self.lock:
            return len(self.frames)

    def get_current_index(self):
        """Get current frame index."""
        with self.lock:
            return self.current_index

    def step_forward(self):
        """Move to next frame."""
        with self.lock:
            if self.current_index < len(self.frames) - 1:
                self.current_index += 1
                return True
        return False

    def step_backward(self):
        """Move to previous frame."""
        with self.lock:
            if self.current_index > 0:
                self.current_index -= 1
                return True
        return False

    def clear(self):
        """Clear all frames."""
        with self.lock:
            self.frames.clear()
            self.current_index = -1


class NetworkReceiver(QThread):
    """Network receiver thread."""
    frame_received = pyqtSignal(dict)
    status_updated = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)

    def __init__(self, port, host='0.0.0.0'):
        super().__init__()
        self.port = port
        self.host = host
        self.is_running = False
        self.socket = None
        self.context = None
        self.received_count = 0
        self.start_time = None
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = None
        self.error = None
        
    def run(self):
        """Main receiver thread."""
        self.is_running = True
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        try:
            # Setup ZeroMQ
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

            self.status_updated.emit(f"✓ Listening on port {self.port}")
            self.connection_changed.emit(True)

            last_fps_update = time.time()

            while self.is_running:
                try:
                    # Send acknowledgment
                    self.socket.send_json({'type': 'request_frame'})

                    # Receive frame data
                    frame_data = self.socket.recv_json()

                    # Process received frame
                    self.received_count += 1
                    current_time = time.time()

                    # Decode image data
                    try:
                        img_bytes = bytes.fromhex(frame_data['image_data'])
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                        if img is not None:
                            # Calculate FPS
                            current_fps = 0
                            if self.last_frame_time:
                                time_diff = current_time - self.last_frame_time
                                if time_diff > 0:
                                    current_fps = 1.0 / time_diff
                            
                            # Update frame data
                            frame_data['image'] = img
                            frame_data['received_time'] = current_time
                            frame_data['fps'] = current_fps
                            frame_data['processing_time'] = frame_data.get('processing_time', 0)
                            
                            # Ensure frame_id exists
                            if 'frame_id' not in frame_data:
                                frame_data['frame_id'] = self.received_count

                            self.last_frame_time = current_time

                            # Emit signal to update GUI
                            self.frame_received.emit(frame_data)

                            # Update status periodically
                            if current_time - last_fps_update > 1.0:
                                elapsed = current_time - self.start_time
                                avg_fps = self.received_count / elapsed if elapsed > 0 else 0

                                status = f"Frame {frame_data['frame_id']:06d} | Total: {self.received_count} | Avg FPS: {avg_fps:.1f} | Current FPS: {current_fps:.1f}"
                                self.status_updated.emit(status)
                                last_fps_update = current_time
                        else:
                            self.status_updated.emit(f"✗ Failed to decode frame {frame_data.get('frame_id', '?')}")

                    except Exception as decode_error:
                        self.status_updated.emit(f"✗ Decode error: {decode_error}")
                        # Still send ACK to keep sender moving

                except zmq.Again:
                    # Timeout - continue loop
                    continue
                except Exception as e:
                    if str(e) != self.error:
                        self.error = str(e)
                        self.status_updated.emit(f"✗ Receive error: {e}")
                        time.sleep(0.1)

        except Exception as e:
            self.status_updated.emit(f"✗ Network error: {e}")
            self.connection_changed.emit(False)
        finally:
            self.cleanup()

    def stop(self):
        """Stop the receiver."""
        self.is_running = False

    def cleanup(self):
        """Cleanup network resources."""
        self.connection_changed.emit(False)
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


class StatisticsWidget(QWidget):
    """Statistics display widget."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()

        # Create LCD displays for statistics
        self.frame_count_lcd = QLCDNumber()
        self.frame_count_lcd.setDigitCount(6)
        self.frame_count_lcd.display(0)

        self.fps_lcd = QLCDNumber()
        self.fps_lcd.setDigitCount(5)
        self.fps_lcd.display(0.0)

        self.processing_time_lcd = QLCDNumber()
        self.processing_time_lcd.setDigitCount(5)
        self.processing_time_lcd.display(0.0)

        # Labels
        layout.addWidget(QLabel("Total Frames:"), 0, 0)
        layout.addWidget(self.frame_count_lcd, 0, 1)

        layout.addWidget(QLabel("FPS:"), 1, 0)
        layout.addWidget(self.fps_lcd, 1, 1)

        layout.addWidget(QLabel("Proc Time (s):"), 2, 0)
        layout.addWidget(self.processing_time_lcd, 2, 1)

        # Connection status
        self.connection_status = QLabel("⚫ Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(QLabel("Status:"), 3, 0)
        layout.addWidget(self.connection_status, 3, 1)

        self.setLayout(layout)

    def update_stats(self, frame_count, fps, processing_time):
        """Update statistics display."""
        self.frame_count_lcd.display(frame_count)
        self.fps_lcd.display(fps)
        self.processing_time_lcd.display(processing_time)

    def set_connection_status(self, connected):
        """Update connection status."""
        if connected:
            self.connection_status.setText("🟢 Connected")
            self.connection_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.connection_status.setText("🔴 Disconnected")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")


class ImageDisplayWidget(QLabel):
    """Custom image display widget with zoom and pan."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.setText("Waiting for frames...")
        self.setScaledContents(False)

        # Zoom and pan state
        self.scale_factor = 1.0
        self.pan_offset = [0, 0]
        self.original_pixmap = None

    def set_image(self, cv_image):
        """Set image from OpenCV format."""
        if cv_image is None:
            return

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Create QImage
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Create pixmap and store original
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update_display()

    def update_display(self):
        """Update display with current zoom and pan."""
        if self.original_pixmap is None:
            return

        # Apply scaling
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def zoom_in(self):
        """Zoom in by 25%."""
        self.scale_factor = min(self.scale_factor * 1.25, 5.0)
        self.update_display()

    def zoom_out(self):
        """Zoom out by 25%."""
        self.scale_factor = max(self.scale_factor / 1.25, 0.1)
        self.update_display()

    def reset_zoom(self):
        """Reset zoom to fit."""
        self.scale_factor = 1.0
        self.pan_offset = [0, 0]
        self.update_display()


class ControlPanel(QWidget):
    """Control panel for navigation and playback."""
    play_pause_clicked = pyqtSignal()
    previous_frame = pyqtSignal()
    next_frame = pyqtSignal()
    frame_seek = pyqtSignal(int)
    live_mode_changed = pyqtSignal(bool)
    clear_buffer = pyqtSignal()
    save_frame = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_playing = True
        self.is_live = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Playback controls
        control_group = QGroupBox("Playback Controls")
        control_layout = QHBoxLayout()

        self.live_checkbox = QCheckBox("Live Mode")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.toggled.connect(self.on_live_mode_changed)

        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.clicked.connect(self.on_play_pause)

        self.prev_btn = QPushButton("⏮ Previous")
        self.prev_btn.clicked.connect(self.previous_frame.emit)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("⏭ Next")
        self.next_btn.clicked.connect(self.next_frame.emit)
        self.next_btn.setEnabled(False)

        control_layout.addWidget(self.live_checkbox)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_group.setLayout(control_layout)

        # Frame navigation
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout()

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_seek)
        self.frame_slider.setEnabled(False)

        self.frame_info = QLabel("Frame: 0 / 0")

        nav_layout.addWidget(self.frame_info)
        nav_layout.addWidget(self.frame_slider)
        nav_group.setLayout(nav_layout)

        # Additional controls
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()

        self.clear_btn = QPushButton("🗑️ Clear Buffer")
        self.clear_btn.clicked.connect(self.clear_buffer.emit)

        self.save_btn = QPushButton("💾 Save Frame")
        self.save_btn.clicked.connect(self.save_frame.emit)

        action_layout.addWidget(self.clear_btn)
        action_layout.addWidget(self.save_btn)
        action_group.setLayout(action_layout)

        layout.addWidget(control_group)
        layout.addWidget(nav_group)
        layout.addWidget(action_group)
        self.setLayout(layout)

    def on_play_pause(self):
        """Handle play/pause button."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
        self.play_pause_clicked.emit()

    def on_live_mode_changed(self, checked):
        """Handle live mode change."""
        self.is_live = checked
        self.prev_btn.setEnabled(not checked)
        self.next_btn.setEnabled(not checked)
        self.frame_slider.setEnabled(not checked)
        
        # If switching to live mode, enable play/pause
        if checked:
            self.play_pause_btn.setEnabled(True)
        
        self.live_mode_changed.emit(checked)

    def on_frame_seek(self, value):
        """Handle frame slider change."""
        if not self.is_live:
            self.frame_seek.emit(value)

    def update_frame_info(self, current, total):
        """Update frame information."""
        self.frame_info.setText(f"Frame: {current + 1} / {total}")
        if total > 0:
            self.frame_slider.setMaximum(total - 1)
            if not self.is_live:
                self.frame_slider.setValue(current)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.frame_buffer = FrameBuffer(max_size=args.buffer_size)
        self.receiver = None
        self.is_live_mode = True
        self.is_paused = False
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)

        self.init_ui()
        self.start_receiver()

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle(f"KITTI 3D Detection Receiver - Port {self.args.port}")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Image display
        self.image_display = ImageDisplayWidget()

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("🔍+ Zoom In")
        zoom_out_btn = QPushButton("🔍- Zoom Out")
        zoom_reset_btn = QPushButton("🔄 Reset Zoom")

        zoom_in_btn.clicked.connect(self.image_display.zoom_in)
        zoom_out_btn.clicked.connect(self.image_display.zoom_out)
        zoom_reset_btn.clicked.connect(self.image_display.reset_zoom)

        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_reset_btn)
        zoom_layout.addStretch()

        left_layout.addWidget(self.image_display)
        left_layout.addLayout(zoom_layout)
        left_panel.setLayout(left_layout)

        # Right panel - Controls and info
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Statistics
        self.stats_widget = StatisticsWidget()

        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.play_pause_clicked.connect(self.on_play_pause)
        self.control_panel.previous_frame.connect(self.on_previous_frame)
        self.control_panel.next_frame.connect(self.on_next_frame)
        self.control_panel.frame_seek.connect(self.on_frame_seek)
        self.control_panel.live_mode_changed.connect(self.on_live_mode_changed)
        self.control_panel.clear_buffer.connect(self.on_clear_buffer)
        self.control_panel.save_frame.connect(self.on_save_frame)

        # Log display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)

        right_layout.addWidget(self.stats_widget)
        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(log_group)
        right_layout.addStretch()
        right_panel.setLayout(right_layout)

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1000, 400])

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Starting receiver...")

        # Start display timer
        self.display_timer.start(33)  # ~30 FPS display update

    def start_receiver(self):
        """Start network receiver."""
        self.receiver = NetworkReceiver(self.args.port, self.args.host)
        self.receiver.frame_received.connect(self.on_frame_received)
        self.receiver.status_updated.connect(self.on_status_updated)
        self.receiver.connection_changed.connect(self.on_connection_changed)
        self.receiver.start()

    @pyqtSlot(dict)
    def on_frame_received(self, frame_data):
        """Handle received frame."""
        self.frame_buffer.add_frame(frame_data)

        # Update statistics
        frame_count = self.frame_buffer.get_frame_count()
        fps = frame_data.get('fps', 0)
        processing_time = frame_data.get('processing_time', 0)

        self.stats_widget.update_stats(frame_count, fps, processing_time)
        
        # Update frame info only if in live mode or if this is the current frame
        if self.is_live_mode:
            self.control_panel.update_frame_info(
                self.frame_buffer.get_current_index(),
                frame_count
            )

    @pyqtSlot(str)
    def on_status_updated(self, message):
        """Handle status update."""
        self.status_bar.showMessage(message)
        self.log_display.append(f"[{time.strftime('%H:%M:%S')}] {message}")

        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(bool)
    def on_connection_changed(self, connected):
        """Handle connection status change."""
        self.stats_widget.set_connection_status(connected)

    def update_display(self):
        """Update image display."""
        if self.is_paused:
            return
            
        frame_data = None
        
        if self.is_live_mode:
            frame_data = self.frame_buffer.get_current_frame()
        else:
            current_idx = self.frame_buffer.get_current_index()
            frame_data = self.frame_buffer.get_frame_at(current_idx)

        if frame_data and 'image' in frame_data:
            self.image_display.set_image(frame_data['image'])

    def on_play_pause(self):
        """Handle play/pause."""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.on_status_updated("Playback paused")
        else:
            self.on_status_updated("Playback resumed")

    def on_previous_frame(self):
        """Go to previous frame."""
        if not self.is_live_mode and self.frame_buffer.step_backward():
            self.control_panel.update_frame_info(
                self.frame_buffer.get_current_index(),
                self.frame_buffer.get_frame_count()
            )
            self.update_display()

    def on_next_frame(self):
        """Go to next frame."""
        if not self.is_live_mode and self.frame_buffer.step_forward():
            self.control_panel.update_frame_info(
                self.frame_buffer.get_current_index(),
                self.frame_buffer.get_frame_count()
            )
            self.update_display()

    def on_frame_seek(self, index):
        """Seek to specific frame."""
        if not self.is_live_mode and self.frame_buffer.set_current_index(index):
            self.control_panel.update_frame_info(
                self.frame_buffer.get_current_index(),
                self.frame_buffer.get_frame_count()
            )
            self.update_display()

    def on_live_mode_changed(self, is_live):
        """Handle live mode change."""
        self.is_live_mode = is_live
        
        if is_live:
            self.on_status_updated("Switched to live mode")
            # Resume playback if paused
            self.is_paused = False
            self.control_panel.play_pause_btn.setText("⏸ Pause")
        else:
            self.on_status_updated("Switched to replay mode")
            # Update frame info for current position
            self.control_panel.update_frame_info(
                self.frame_buffer.get_current_index(),
                self.frame_buffer.get_frame_count()
            )

    def on_clear_buffer(self):
        """Clear frame buffer."""
        self.frame_buffer.clear()
        self.control_panel.update_frame_info(0, 0)
        self.image_display.setText("Buffer cleared - waiting for frames...")
        self.on_status_updated("Frame buffer cleared")

    def on_save_frame(self):
        """Save current frame."""
        current_frame = None
        
        if self.is_live_mode:
            current_frame = self.frame_buffer.get_current_frame()
        else:
            current_idx = self.frame_buffer.get_current_index()
            current_frame = self.frame_buffer.get_frame_at(current_idx)

        if current_frame and 'image' in current_frame:
            # Open file dialog to choose save location
            frame_id = current_frame.get('frame_id', 'unknown')
            timestamp = int(time.time())
            default_filename = f"kitti_frame_{frame_id}_{timestamp}.jpg"
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Frame",
                default_filename,
                "JPEG files (*.jpg);;PNG files (*.png);;All files (*.*)"
            )
            
            if filename:
                try:
                    cv2.imwrite(filename, current_frame['image'])
                    self.on_status_updated(f"Frame saved as {os.path.basename(filename)}")
                    QMessageBox.information(self, "Success", f"Frame saved successfully:\n{filename}")
                except Exception as e:
                    self.on_status_updated(f"Error saving frame: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to save frame:\n{e}")
        else:
            self.on_status_updated("No frame to save")
            QMessageBox.warning(self, "Warning", "No frame available to save")

    def closeEvent(self, event):
        """Handle window close."""
        if self.receiver:
            self.receiver.stop()
            self.receiver.wait()
        event.accept()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='KITTI 3D Detection Receiver')
    parser.add_argument('--port', type=int, default=5555, help='Listening port')
    parser.add_argument('--host', default='0.0.0.0', help='Binding host (* for all interfaces)')
    parser.add_argument('--buffer-size', type=int, default=100, help='Frame buffer size')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print('=' * 60)
    print('KITTI 3D Detection Receiver - Real-time Display')
    print('=' * 60)
    print(f'Listening on: {args.host}:{args.port}')
    print(f'Buffer size: {args.buffer_size} frames')
    print('-' * 60)

    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern looking style

    # Set dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Create and show main window
    main_window = MainWindow(args)
    main_window.show()

    # Run application
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")


if __name__ == '__main__':
    main()