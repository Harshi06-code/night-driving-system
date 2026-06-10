import cv2
import numpy as np
from module2_glare import detect_and_reduce_glare

VIDEO_PATH = "videos/test_night.mp4"

def enhance_lanes(frame):
    height, width = frame.shape[:2]

    # Only work on bottom 35% of frame — that is where road is
    roi_top = int(height * 0.65)
    roi = frame[roi_top:height, 0:width]

    # Convert ROI to grayscale and blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 30, 90)

    # Hough lines on small ROI only
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=50,
        maxLineGap=120
    )

    lane_frame = frame.copy()

    if lines is not None:
        left_lines = []
        right_lines = []
        mid_x = width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope — ignore nearly horizontal lines
            if abs(slope) < 0.3:
                continue

            # Left side of frame = left lane
            if x1 < mid_x and x2 < mid_x and slope < 0:
                left_lines.append((x1, y1 + roi_top,
                                   x2, y2 + roi_top))
            # Right side of frame = right lane
            elif x1 > mid_x and x2 > mid_x and slope > 0:
                right_lines.append((x1, y1 + roi_top,
                                    x2, y2 + roi_top))

        # Draw single averaged left line
        if left_lines:
            x1 = int(np.mean([l[0] for l in left_lines]))
            y1 = int(np.mean([l[1] for l in left_lines]))
            x2 = int(np.mean([l[2] for l in left_lines]))
            y2 = int(np.mean([l[3] for l in left_lines]))
            cv2.line(lane_frame, (x1, y1), (x2, y2),
                     (0, 255, 0), 4)

        # Draw single averaged right line
        if right_lines:
            x1 = int(np.mean([l[0] for l in right_lines]))
            y1 = int(np.mean([l[1] for l in right_lines]))
            x2 = int(np.mean([l[2] for l in right_lines]))
            y2 = int(np.mean([l[3] for l in right_lines]))
            cv2.line(lane_frame, (x1, y1), (x2, y2),
                     (0, 255, 0), 4)

    return lane_frame

def run_lane_module():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Module 3 running — Lane Enhancement")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # First apply glare reduction from Module 2
        glare_reduced, _ = detect_and_reduce_glare(frame)

        # Then enhance lanes on the glare reduced frame
        lane_enhanced = enhance_lanes(glare_reduced)

        # Show all three stages
        # Resize each panel to smaller width so all 3 fit on screen
        h, w = frame.shape[:2]
        small_w = 480
        small_h = int(h * small_w / w)
        
        panel1 = cv2.resize(frame, (small_w, small_h))
        panel2 = cv2.resize(glare_reduced, (small_w, small_h))
        panel3 = cv2.resize(lane_enhanced, (small_w, small_h))
        
        combined = np.hstack((panel1, panel2, panel3))
        cv2.imshow("Original | Glare Reduced | Lane Enhanced", combined)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lane_module()