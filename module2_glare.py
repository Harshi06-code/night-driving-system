import cv2
import numpy as np

VIDEO_PATH = "videos/test_night.mp4"

def detect_and_reduce_glare(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask 1 — pure white bright light
    lower1 = np.array([0, 0, 200])
    upper1 = np.array([180, 40, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Mask 2 — yellowish white headlights
    lower2 = np.array([15, 30, 180])
    upper2 = np.array([35, 180, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Combine both masks
    glare_mask = cv2.bitwise_or(mask1, mask2)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_DILATE, kernel)

    # ✅ Gentle blend dimming — NOT blackout
    result = frame.copy()
    glare_area = result[glare_mask == 255].astype(np.float32)
    dimmed = glare_area * 0.4
    result[glare_mask == 255] = np.clip(dimmed, 0, 255).astype(np.uint8)

    # Draw boxes only around real headlight sized glare
    contours, _ = cv2.findContours(
        glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 800 < area < 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 250 and h < 250:
                cv2.rectangle(result, (x, y), (x+w, y+h),
                              (0, 0, 255), 2)
                cv2.putText(result, "GLARE", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)

    return result, glare_mask

def run_glare_module():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Module 2 running — Glare Detection & Reduction")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed, mask = detect_and_reduce_glare(frame)

        combined = np.hstack((frame, processed))
        cv2.imshow("Original  |  Glare Reduced", combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_glare_module()