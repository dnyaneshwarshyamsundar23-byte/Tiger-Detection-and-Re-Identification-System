"""
Tiger Detection Web UI using Flask
- Live video feed in browser
- Start/Stop button to control detection
- Tiger list showing ID, arrival time, and departure time
"""

import os
import cv2
import torch
import numpy as np
import time
import threading
from datetime import datetime
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import base64
import csv
from io import StringIO, BytesIO

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
CORS(app)
# ----------------------------
# Global Variables
# ----------------------------
tiger_db = {}
tiger_log = {}
tiger_count = 0
is_running = False
current_frame = None
detection_thread = None

# Camera and models
cap = None
yolo = None
resnet = None
transform = None
device = "cuda" if torch.cuda.is_available() else "cpu"

SIMILARITY_THRESHOLD = 0.85

# Create directory for tiger images
TIGER_IMAGES_DIR = "tiger_captures"
os.makedirs(TIGER_IMAGES_DIR, exist_ok=True)

# ----------------------------
# Initialize Models
# ----------------------------
def initialize_models():
    global yolo, resnet, transform, device
    
    print("Loading models...")
    try:
        # Load YOLO
        if os.path.exists("best_yolov8.pt"):
            yolo = YOLO("best_yolov8.pt")
        elif os.path.exists("best_enlightengan_and_yolov8.pt"):
            yolo = YOLO("best_enlightengan_and_yolov8.pt")
        else:
            print("Warning: YOLO model not found!")
            return False
        
        # Load ResNet for embeddings
        resnet = models.resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()
        resnet = resnet.to(device).eval()
        
        transform = transforms.Compose([
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

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(bgr_crop):
    """Extract normalized embedding from a cropped tiger image."""
    try:
        img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(tensor).cpu().numpy().reshape(-1)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(2048)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def frame_to_base64(frame):
    """Convert frame to base64 for sending to browser."""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode()
    return frame_base64

def detection_loop():
    """Main detection loop running in background thread."""
    global current_frame, tiger_db, tiger_count, tiger_log, is_running, cap
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = {}
        
        try:
            results = yolo(frame, conf=0.5)
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if it's a tiger
                    if hasattr(yolo, 'names') and yolo.names[cls_id] == "tiger" and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                            crop = frame[y1:y2, x1:x2]
                            
                            if crop.size > 0:
                                emb = get_embedding(crop)
                                
                                # Match with database
                                if tiger_db:
                                    similarities = {tid: cosine_similarity(emb, db["embedding"]) 
                                                   for tid, db in tiger_db.items()}
                                    best_tid = max(similarities, key=similarities.get)
                                    best_score = similarities[best_tid]
                                    
                                    if best_score > SIMILARITY_THRESHOLD:
                                        tiger_id = best_tid
                                        tiger_db[tiger_id]["last_seen"] = time.time()
                                    else:
                                        tiger_count += 1
                                        tiger_id = f"Tiger_{tiger_count}"
                                        tiger_db[tiger_id] = {
                                            "embedding": emb,
                                            "first_seen": time.time(),
                                            "last_seen": time.time(),
                                            "images": []
                                        }
                                else:
                                    tiger_count += 1
                                    tiger_id = f"Tiger_{tiger_count}"
                                    tiger_db[tiger_id] = {
                                        "embedding": emb,
                                        "first_seen": time.time(),
                                        "last_seen": time.time(),
                                        "images": []
                                    }
                                
                                # Save crop image
                                try:
                                    tiger_dir = os.path.join(TIGER_IMAGES_DIR, tiger_id)
                                    os.makedirs(tiger_dir, exist_ok=True)
                                    img_filename = f"{tiger_id}_{int(time.time()*1000)}.jpg"
                                    img_path = os.path.join(tiger_dir, img_filename)
                                    cv2.imwrite(img_path, crop)
                                    tiger_db[tiger_id]["images"].append(img_path)
                                except Exception as e:
                                    print(f"Error saving tiger image: {e}")
                                
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
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{tiger_id} ({conf:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update tiger log
        for tiger_id in detections:
            if tiger_id not in tiger_log:
                tiger_log[tiger_id] = {
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "departed": False
                }
            else:
                tiger_log[tiger_id]["last_seen"] = time.time()
                if tiger_log[tiger_id]["departed"]:
                    tiger_log[tiger_id]["departed"] = False
        
        # Mark tigers as departed
        current_ids = set(detections.keys())
        for tiger_id in tiger_log:
            if tiger_id not in current_ids and not tiger_log[tiger_id]["departed"]:
                tiger_log[tiger_id]["departed"] = True
        
        # Store frame
        current_frame = annotated_frame
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)
    
    if cap:
        cap.release()

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start the detection."""
    global is_running, detection_thread
    
    if not is_running:
        is_running = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        return jsonify({"status": "success", "message": "Detection started"})
    
    return jsonify({"status": "error", "message": "Detection already running"})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop the detection."""
    global is_running, cap
    
    if is_running:
        is_running = False
        if cap:
            cap.release()
        return jsonify({"status": "success", "message": "Detection stopped"})
    
    return jsonify({"status": "error", "message": "Detection not running"})

@app.route('/api/frame')
def get_frame():
    """Get the current frame as base64."""
    global current_frame
    
    if current_frame is None:
        return jsonify({"frame": None, "status": "no_frame"})
    
    frame_data = frame_to_base64(current_frame)
    return jsonify({"frame": frame_data, "status": "ok"})

@app.route('/api/tigers')
def get_tigers():
    """Get all detected tigers."""
    tigers_list = []
    
    for tiger_id, log_info in sorted(tiger_log.items()):
        arrival_time = datetime.fromtimestamp(log_info["first_seen"]).strftime("%H:%M:%S")
        
        if log_info["departed"]:
            departure_time = datetime.fromtimestamp(log_info["last_seen"]).strftime("%H:%M:%S")
            status = "Departed"
        else:
            departure_time = "Active"
            status = "Present"
        
        tigers_list.append({
            "tiger_id": tiger_id,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "status": status
        })
    
    return jsonify({"tigers": tigers_list, "is_running": is_running})

@app.route('/api/status')
def get_status():
    """Get current status."""
    return jsonify({"is_running": is_running, "tiger_count": len(tiger_log)})

@app.route('/api/download-csv')
def download_csv():
    """Download tiger detections as CSV."""
    # Create CSV in memory
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Write header
    csv_writer.writerow(['Tiger ID', 'Arrival Time', 'Departure Time', 'Status', 'Duration (seconds)'])
    
    # Write data
    for tiger_id, log_info in sorted(tiger_log.items()):
        arrival_time = datetime.fromtimestamp(log_info["first_seen"]).strftime("%Y-%m-%d %H:%M:%S")
        
        if log_info["departed"]:
            departure_time = datetime.fromtimestamp(log_info["last_seen"]).strftime("%Y-%m-%d %H:%M:%S")
            status = "Departed"
            duration = int(log_info["last_seen"] - log_info["first_seen"])
        else:
            departure_time = "Active"
            status = "Present"
            duration = int(time.time() - log_info["first_seen"])
        
        csv_writer.writerow([tiger_id, arrival_time, departure_time, status, duration])
    
    # Convert to bytes
    csv_buffer.seek(0)
    csv_bytes = BytesIO(csv_buffer.getvalue().encode('utf-8'))
    
    # Return as file download
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"tiger_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Initialize models
    if not initialize_models():
        print("Failed to initialize models")
        exit(1)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run Flask app
    print("Starting Tiger Detection Web UI...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
