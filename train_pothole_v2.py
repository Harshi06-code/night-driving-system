from ultralytics import YOLO

# Use small model instead of nano
model = YOLO("yolov8s.pt")

results = model.train(
    data="pothole_combined/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,          # smaller batch because bigger model
    name="pothole_v2_model",
    patience=15,
    device="cpu",
    hsv_v=0.5,        # more brightness variation
    hsv_s=0.5,
    mosaic=1.0,
    fliplr=0.5,
)

print("V2 model saved at: runs/detect/pothole_v2_model/weights/best.pt")