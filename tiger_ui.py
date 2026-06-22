"""
Tiger Detection UI with Live Feed and Detection Tracking
- Live video feed displayed in one corner
- Start/Stop button to control detection
- Tiger list showing ID, arrival time, and departure time
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QGridLayout, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QHeaderView

# ----------------------------
# Detection Thread
# ----------------------------
class DetectionThread(QThread):
    frame_ready = pyqtSignal(object)  # emits (frame, detections_dict)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.running = False
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        self.resnet = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def setup_models(self):
        """Initialize YOLO and ResNet models."""
        print("Loading models...")
        try:
            # Load YOLO
            if os.path.exists("best_yolov8.pt"):
                self.model = YOLO("best_yolov8.pt")
            elif os.path.exists("best_enlightengan_and_yolov8.pt"):
                self.model = YOLO("best_enlightengan_and_yolov8.pt")
            else:
                print("Warning: YOLO model not found!")
                return False
            
            # Load ResNet for embeddings
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.fc = torch.nn.Identity()
            self.resnet = self.resnet.to(self.device).eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_embedding(self, bgr_crop):
        """Extract normalized embedding from a cropped tiger image."""
        try:
            img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.resnet(tensor).cpu().numpy().reshape(-1)
            norm = np.linalg.norm(emb)
            if norm == 0:
                return emb
            return emb / norm
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(2048)
    
    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        a = a.reshape(-1)
        b = b.reshape(-1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    
    def run(self):
        """Main detection loop."""
        if not self.setup_models():
            print("Failed to setup models")
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_index}")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        tiger_db = {}  # tiger_id -> {"embedding": emb, "first_seen": time}
        tiger_count = 0
        SIMILARITY_THRESHOLD = 0.85
        current_detections = {}  # Currently visible tigers in frame
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            detections = {}  # {tiger_id: {"bbox": (x1,y1,x2,y2), "confidence": conf}}
            
            try:
                results = self.model(frame, conf=0.5)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Check if it's a tiger
                        if hasattr(self.model, 'names') and self.model.names[cls_id] == "tiger" and conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Ensure crop is valid
                            if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                                crop = frame[y1:y2, x1:x2]
                                
                                if crop.size > 0:
                                    # Get embedding
                                    emb = self.get_embedding(crop)
                                    
                                    # Match with database
                                    if tiger_db:
                                        similarities = {tid: self.cosine_similarity(emb, db["embedding"]) 
                                                       for tid, db in tiger_db.items()}
                                        best_tid = max(similarities, key=similarities.get)
                                        best_score = similarities[best_tid]
                                        
                                        if best_score > SIMILARITY_THRESHOLD:
                                            tiger_id = best_tid
                                            # Update last seen time
                                            tiger_db[tiger_id]["last_seen"] = time.time()
                                        else:
                                            # New tiger
                                            tiger_count += 1
                                            tiger_id = f"Tiger_{tiger_count}"
                                            tiger_db[tiger_id] = {
                                                "embedding": emb,
                                                "first_seen": time.time(),
                                                "last_seen": time.time()
                                            }
                                    else:
                                        # First tiger
                                        tiger_count += 1
                                        tiger_id = f"Tiger_{tiger_count}"
                                        tiger_db[tiger_id] = {
                                            "embedding": emb,
                                            "first_seen": time.time(),
                                            "last_seen": time.time()
                                        }
                                    
                                    detections[tiger_id] = {
                                        "bbox": (x1, y1, x2, y2),
                                        "confidence": conf
                                    }
            except Exception as e:
                print(f"Error in detection: {e}")
            
            # Draw detections on frame
            annotated_frame = frame.copy()
            for tiger_id, detect_info in detections.items():
                x1, y1, x2, y2 = detect_info["bbox"]
                conf = detect_info["confidence"]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                label = f"{tiger_id} ({conf:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Emit frame and detection data
            self.frame_ready.emit((annotated_frame, detections, tiger_db))
    
    def stop(self):
        """Stop the detection thread."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


# ----------------------------
# Main UI Window
# ----------------------------
class TigerDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiger Detection System")
        self.setGeometry(100, 100, 1400, 800)
        
        # Detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_ready.connect(self.update_frame)
        
        # Tiger tracking
        self.tiger_log = {}  # {tiger_id: {"first_seen": time, "last_seen": time, "departed": bool}}
        self.current_frame = None
        self.is_running = False
        
        # Setup UI
        self.setup_ui()
        
        # Timer for update
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_table)
        self.update_timer.start(1000)  # Update every second
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout(main_widget)
        
        # Left side: Video feed
        left_layout = QVBoxLayout()
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.start_btn.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        # Right side: Tiger list
        right_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Tiger Detections")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(title)
        
        # Table for tiger detections
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Tiger ID", "Arrival Time", "Departure Time", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        right_layout.addWidget(self.table)
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: red; padding: 10px;")
        right_layout.addWidget(self.status_label)
        
        # Add layouts to main
        main_layout.addLayout(left_layout, 3)  # 60% width
        main_layout.addLayout(right_layout, 2)  # 40% width
        
        main_widget.setLayout(main_layout)
    
    def start_detection(self):
        """Start the detection."""
        if not self.is_running:
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("color: green; padding: 10px; font-weight: bold;")
            self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the detection."""
        if self.is_running:
            self.is_running = False
            self.detection_thread.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Status: Stopped")
            self.status_label.setStyleSheet("color: red; padding: 10px; font-weight: bold;")
            self.video_label.clear()
    
    def update_frame(self, data):
        """Update video frame and handle detections."""
        frame, detections, tiger_db = data
        
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        
        # Resize for display
        display_size = (640, 480)
        frame_rgb = cv2.resize(frame_rgb, display_size)
        
        # Convert to QImage
        bytes_per_line = 3 * display_size[0]
        qt_image = QImage(frame_rgb.data, display_size[0], display_size[1], 
                         bytes_per_line, QImage.Format_RGB888)
        
        # Update label
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
        # Update tiger log
        for tiger_id, detect_info in detections.items():
            if tiger_id not in self.tiger_log:
                self.tiger_log[tiger_id] = {
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "departed": False
                }
            else:
                self.tiger_log[tiger_id]["last_seen"] = time.time()
                if self.tiger_log[tiger_id]["departed"]:
                    self.tiger_log[tiger_id]["departed"] = False
        
        # Mark tigers as departed if not in current frame
        current_ids = set(detections.keys())
        for tiger_id in self.tiger_log:
            if tiger_id not in current_ids and not self.tiger_log[tiger_id]["departed"]:
                self.tiger_log[tiger_id]["departed"] = True
    
    def refresh_table(self):
        """Refresh the tiger detection table."""
        self.table.setRowCount(0)
        
        for tiger_id, log_info in sorted(self.tiger_log.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Tiger ID
            item = QTableWidgetItem(tiger_id)
            item.setFont(QFont("Arial", 10))
            self.table.setItem(row, 0, item)
            
            # Arrival time
            arrival_time = datetime.fromtimestamp(log_info["first_seen"]).strftime("%H:%M:%S")
            item = QTableWidgetItem(arrival_time)
            self.table.setItem(row, 1, item)
            
            # Departure time
            if log_info["departed"]:
                departure_time = datetime.fromtimestamp(log_info["last_seen"]).strftime("%H:%M:%S")
            else:
                departure_time = "Active"
            item = QTableWidgetItem(departure_time)
            self.table.setItem(row, 2, item)
            
            # Status
            status = "Present" if not log_info["departed"] else "Departed"
            item = QTableWidgetItem(status)
            if not log_info["departed"]:
                item.setBackground(QColor(144, 238, 144))  # Light green
            else:
                item.setBackground(QColor(255, 200, 200))  # Light red
            self.table.setItem(row, 3, item)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_running:
            self.stop_detection()
        event.accept()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TigerDetectionUI()
    window.show()
    sys.exit(app.exec_())
