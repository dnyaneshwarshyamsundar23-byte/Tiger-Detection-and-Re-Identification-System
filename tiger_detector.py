import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load YOLO model
model = YOLO("best_enlightengan_and_yolov8.pt")

# Load ResNet50 for stripe embeddings
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Database for tiger embeddings
tiger_db = {}  # { "Tiger_1": embedding_vector, ... }
tiger_count = 0


def get_embedding(image):
    """Extract embedding from a cropped tiger image."""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(img_t).numpy()
    return emb


def identify_tiger(embedding, threshold=0.80):
    """Check if tiger is new or known based on stripe similarity."""
    global tiger_count, tiger_db
    if not tiger_db:
        tiger_count += 1
        tiger_id = f"Tiger_{tiger_count}"
        tiger_db[tiger_id] = embedding
        return tiger_id, True

    similarities = [cosine_similarity(embedding, db_emb)[0][0] for db_emb in tiger_db.values()]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        tiger_id = list(tiger_db.keys())[best_idx]
        return tiger_id, False  # Existing tiger
    else:
        tiger_count += 1
        tiger_id = f"Tiger_{tiger_count}"
        tiger_db[tiger_id] = embedding
        return tiger_id, True


# Start webcam
cap = cv2.VideoCapture(0)

os.makedirs("tiger_database", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if model.names[cls_id] == "tiger" and conf > 0.80:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                # Extract stripe embedding
                emb = get_embedding(crop)

                # Identify tiger
                tiger_id, is_new = identify_tiger(emb)

                # Save only if it's a new tiger
                if is_new:
                    cv2.imwrite(f"tiger_database/{tiger_id}.jpg", crop)

                # Show label on screen
                cv2.putText(frame, tiger_id, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Tiger Re-Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
