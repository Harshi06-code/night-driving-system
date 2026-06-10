from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/pothole_night_model/weights/best.pt")

# Test on a single frame
cap = cv2.VideoCapture("videos/test_night.mp4")
ret, frame = cap.read()
cap.release()

results = model(frame, verbose=True)[0]

print(f"Total detections: {len(results.boxes)}")
for box in results.boxes:
    print(f"Confidence: {float(box.conf[0]):.2f}")