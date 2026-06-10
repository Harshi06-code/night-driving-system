from ultralytics import YOLO

print("=" * 40)
print("  Pothole Detection Model Training")
print("  Dataset: 5839 images (day + night)")
print("  This will take 1-3 hours on CPU")
print("  Do NOT close this window!")
print("=" * 40)

# Load base YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on combined pothole dataset
results = model.train(
    data="pothole_combined/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="pothole_night_model",
    patience=15,
    device="cpu",

    # Augmentations to simulate different lighting
    hsv_v=0.4,      # brightness variation — simulates night/day
    hsv_s=0.5,      # saturation variation
    hsv_h=0.015,    # hue variation
    fliplr=0.5,     # horizontal flip
    mosaic=1.0,     # mosaic augmentation
    degrees=5.0,    # slight rotation
)

print("\n" + "=" * 40)
print("Training complete!")
print("Best model saved at:")
print("runs/detect/pothole_night_model/weights/best.pt")
print("=" * 40)

