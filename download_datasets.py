import roboflow

# This will ask you to login once in browser — free account needed
roboflow.login()

rf = roboflow.Roboflow()

print("Downloading Dataset 1 — Large pothole dataset...")
proj1 = rf.workspace("pothole-detection").project("pothole-detection-system")
proj1.version(1).download("yolov8", location="pothole_dataset_1")

print("Downloading Dataset 2 — Pothole + hump + vehicle...")
proj2 = rf.workspace("aegis").project("pothole-detection")
proj2.version(1).download("yolov8", location="pothole_dataset_2")

print("Downloading Dataset 3 — Kartik pothole...")
proj3 = rf.workspace("kartik-zvust").project("pothole-detection-yolo-v8")
proj3.version(1).download("yolov8", location="pothole_dataset_3")

print("All 3 datasets downloaded successfully!")