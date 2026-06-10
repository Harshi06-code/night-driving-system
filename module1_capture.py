import cv2

# ── Change this to your actual video filename ──
VIDEO_PATH = "videos/test_night.mp4"

def run_capture():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        print("Check that the file exists at:", VIDEO_PATH)
        return

    print("Video loaded successfully!")
    print("Press Q to quit the window.")

    while True:
        ret, frame = cap.read()

        # If video ends, loop back to start
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Show the frame in a window
        cv2.imshow("Night Driving - Module 1", frame)

        # Press Q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video closed.")

# Run itq
if __name__ == "__main__":
    run_capture()