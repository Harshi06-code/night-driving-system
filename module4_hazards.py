import cv2
import numpy as np
from ultralytics import YOLO
from module2_glare import detect_and_reduce_glare
from module3_lanes import enhance_lanes

VIDEO_PATH = "videos/test_night.mp4"
model = YOLO("yolov8n.pt")

HAZARD_CLASSES = {
    0: "Person", 1: "Bicycle", 2: "Car",
    3: "Motorcycle", 5: "Bus", 7: "Truck",
    9: "Traffic Light", 11: "Stop Sign"
}
def draw_status_bar(frame, glare_count, hazard_count, lane_active):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Split bar into 4 equal sections
    section = w // 4

    # Section 1 — System active
    cv2.circle(frame, (20, 32), 7, (0, 200, 0), -1)
    cv2.putText(frame, "SYSTEM ACTIVE", (35, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    # Section 2 — Glare
    glare_color = (0, 100, 255) if glare_count > 0 else (0, 200, 0)
    glare_text = f"GLARE: {glare_count}" if glare_count > 0 else "GLARE: CLEAR"
    cv2.circle(frame, (section + 10, 25), 6, glare_color, -1)
    cv2.putText(frame, glare_text, (section + 22, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, glare_color, 1)

    # Section 3 — Hazard
    if hazard_count > 0:
        cv2.circle(frame, (section * 2 + 10, 25), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"HAZARD: {hazard_count}",
                    (section * 2 + 22, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    else:
        cv2.circle(frame, (section * 2 + 10, 25), 6, (0, 200, 0), -1)
        cv2.putText(frame, "HAZARD: NONE",
                    (section * 2 + 22, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

    # Section 4 — Lane
    lane_color = (0, 200, 0) if lane_active else (100, 100, 100)
    cv2.circle(frame, (section * 3 + 10, 25), 6, lane_color, -1)
    cv2.putText(frame, "LANE: ON" if lane_active else "LANE: OFF",
                (section * 3 + 22, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, lane_color, 1)

    return frame

def draw_bottom_bar(frame, glare_count, hazard_count):
    h, w = frame.shape[:2]

    # Dark bottom bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Glare count
    cv2.putText(frame, "GLARE ZONES", (20, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, str(glare_count), (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hazard count
    hazard_color = (0, 0, 255) if hazard_count > 0 else (255, 255, 255)
    hazard_val = f"{hazard_count} DETECTED" if hazard_count > 0 else "0"
    cv2.putText(frame, "HAZARDS", (160, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, hazard_val, (160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hazard_color, 2)

    # Lane tracking
    cv2.putText(frame, "LANE STATUS", (380, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "TRACKING", (380, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    return frame

def detect_hazards_clean(frame):
    result_frame = frame.copy()
    results = model(frame, verbose=False)[0]

    hazard_count = 0

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id in HAZARD_CLASSES and confidence > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = HAZARD_CLASSES[class_id]
            hazard_count += 1

            # Clean bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 220), 2)

            # Clean label background
            label_text = f"{label} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result_frame,
                          (x1, y1 - th - 8), (x1 + tw + 6, y1),
                          (0, 0, 220), -1)
            cv2.putText(result_frame, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return result_frame, hazard_count

def run_hazard_module():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Full system running...")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Full pipeline
        glare_reduced, glare_mask = detect_and_reduce_glare(frame)
        lane_enhanced = enhance_lanes(glare_reduced)
        final_output, hazard_count = detect_hazards_clean(lane_enhanced)

        # Count glare zones
        contours, _ = cv2.findContours(glare_mask,
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glare_count = sum(1 for c in contours if cv2.contourArea(c) > 500)

        # Add clean UI bars
        final_output = draw_status_bar(final_output, glare_count,
                                       hazard_count, True)
        final_output = draw_bottom_bar(final_output, glare_count, hazard_count)

        # Resize both to same height
        target_h = 400
        target_w = int(frame.shape[1] * target_h / frame.shape[0])

        left = cv2.resize(frame, (target_w, target_h))
        right = cv2.resize(final_output, (target_w, target_h))

        # Add panel labels
        cv2.putText(left, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(right, "AI PROCESSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        # Thin divider line between panels
        divider = np.zeros((target_h, 3, 3), dtype=np.uint8)
        divider[:] = (60, 60, 60)

        combined = np.hstack((left, divider, right))
        cv2.imshow("Smart Night Driving System", combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hazard_module()