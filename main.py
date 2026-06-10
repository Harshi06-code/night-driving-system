import cv2
import numpy as np
from ultralytics import YOLO
from module2_glare import detect_and_reduce_glare
from module3_lanes import enhance_lanes

# ── Configuration ──────────────────────────────
VIDEO_PATH = "videos/test_night.mp4"

# ── Load Models Once ───────────────────────────
print("Loading models...")
pothole_model = YOLO("runs/detect/pothole_v2_model/weights/best.pt")
vehicle_model = YOLO("yolov8n.pt")
print("Models loaded!")

# ── Class Definitions ──────────────────────────
VEHICLE_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    9: "Traffic Light",
    11: "Stop Sign"
}

# ── Hazard Detection ───────────────────────────
def detect_hazards(display_frame, detection_frame):
    result_frame = display_frame.copy()
    hazard_count = 0
    frame_height = detection_frame.shape[0]

    # Pothole detection on ORIGINAL frame — better accuracy
    pothole_results = pothole_model(detection_frame, verbose=False)[0]
    for box in pothole_results.boxes:
        confidence = float(box.conf[0])
        if confidence > 0.20:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if y1 < frame_height * 0.40:
                continue

            label = f"Pothole {confidence:.0%}"
            hazard_count += 1

            cv2.rectangle(result_frame,
                          (x1, y1), (x2, y2), (0, 165, 255), 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result_frame,
                          (x1, y1 - th - 8),
                          (x1 + tw + 6, y1),
                          (0, 165, 255), -1)
            cv2.putText(result_frame, label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)

    # Vehicle detection on original frame
    vehicle_results = vehicle_model(detection_frame, verbose=False)[0]
    for box in vehicle_results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id in VEHICLE_CLASSES and confidence > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{VEHICLE_CLASSES[class_id]} {confidence:.0%}"
            hazard_count += 1

            cv2.rectangle(result_frame,
                          (x1, y1), (x2, y2), (0, 0, 220), 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result_frame,
                          (x1, y1 - th - 8),
                          (x1 + tw + 6, y1),
                          (0, 0, 220), -1)
            cv2.putText(result_frame, label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)

    return result_frame, hazard_count

# ── Status Bar Top ─────────────────────────────
def draw_status_bar(frame, glare_count, hazard_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    section = w // 4

    # System active
    cv2.circle(frame, (18, 28), 7, (0, 210, 0), -1)
    cv2.putText(frame, "SYSTEM ACTIVE", (30, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1)

    # Glare status
    g_color = (0, 120, 255) if glare_count > 0 else (0, 210, 0)
    g_text = f"GLARE: {glare_count}" if glare_count > 0 else "GLARE: CLEAR"
    cv2.circle(frame, (section + 10, 28), 7, g_color, -1)
    cv2.putText(frame, g_text, (section + 24, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 1)

    # Hazard status
    h_color = (0, 0, 255) if hazard_count > 0 else (0, 210, 0)
    h_text = f"HAZARD: {hazard_count}" if hazard_count > 0 else "HAZARD: NONE"
    cv2.circle(frame, (section * 2 + 10, 28), 7, h_color, -1)
    cv2.putText(frame, h_text, (section * 2 + 24, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1)

    # Lane status
    cv2.circle(frame, (section * 3 + 10, 28), 7, (0, 210, 0), -1)
    cv2.putText(frame, "LANE: ON", (section * 3 + 24, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1)

    return frame

# ── Bottom Info Bar ────────────────────────────
def draw_bottom_bar(frame, glare_count, hazard_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 52), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Glare zones
    cv2.putText(frame, "GLARE ZONES", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, str(glare_count), (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 120, 255) if glare_count > 0 else (255, 255, 255), 2)

    # Hazards
    cv2.putText(frame, "HAZARDS", (w // 3, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    h_val = f"{hazard_count} DETECTED" if hazard_count > 0 else "0"
    h_col = (0, 0, 255) if hazard_count > 0 else (255, 255, 255)
    cv2.putText(frame, h_val, (w // 3, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, h_col, 2)

    # Lane status
    cv2.putText(frame, "LANE STATUS", (w * 2 // 3, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "TRACKING", (w * 2 // 3, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2)

    return frame

# ── Main Pipeline ──────────────────────────────
def run():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("=" * 40)
    print("  Smart Night Driving System")
    print("  All modules loaded successfully")
    print("  Press Q to quit")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        original = frame.copy()

        # Step 1 — Glare reduction
        glare_reduced, glare_mask = detect_and_reduce_glare(frame)

        # Step 2 — Lane enhancement
        lane_enhanced = enhance_lanes(glare_reduced)

        # Step 3 — Hazard detection
        # display_frame = lane_enhanced (with all processing)
        # detection_frame = original frame (better for model accuracy)
        final_output, hazard_count = detect_hazards(lane_enhanced, frame)

        # Step 4 — Count glare zones
        contours, _ = cv2.findContours(
            glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glare_count = sum(
            1 for c in contours if cv2.contourArea(c) > 500)

        # Step 5 — Draw UI
        final_output = draw_status_bar(
            final_output, glare_count, hazard_count)
        final_output = draw_bottom_bar(
            final_output, glare_count, hazard_count)

        # Step 6 — Combine panels
        target_h = 420
        target_w = int(original.shape[1] * target_h / original.shape[0])

        left  = cv2.resize(original, (target_w, target_h))
        right = cv2.resize(final_output, (target_w, target_h))

        cv2.putText(left, "ORIGINAL", (12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2)
        cv2.putText(right, "AI PROCESSED", (12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 210, 0), 2)

        divider = np.full((target_h, 4, 3), 60, dtype=np.uint8)
        combined = np.hstack((left, divider, right))

        cv2.imshow("Smart Night Driving System", combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System stopped.")

if __name__ == "__main__":
    run()