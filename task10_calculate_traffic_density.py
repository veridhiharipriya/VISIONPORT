
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 10")
print("TRAFFIC DENSITY CALCULATION")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

print("Traffic video opened successfully!")

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

# Density thresholds (percentage of frame occupied)
LOW_LIMIT = 5
MEDIUM_LIMIT = 15

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]
    total_frame_area = height * width

    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.40,
        verbose=False
    )

    total_vehicle_area = 0
    vehicle_count = 0

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            # Calculate bounding box area
            vehicle_area = max(0, x2 - x1) * max(0, y2 - y1)

            total_vehicle_area += vehicle_area
            vehicle_count += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # Calculate occupied frame percentage
    density_percentage = (
        total_vehicle_area / total_frame_area
    ) * 100

    # Classify traffic density
    if density_percentage < LOW_LIMIT:
        density = "LOW"
    elif density_percentage < MEDIUM_LIMIT:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # Display density information
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Density: {density}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Occupied Area: {density_percentage:.2f}%",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("VisionPort - Traffic Density", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print current density
    print(
        f"Vehicles: {vehicle_count} | "
        f"Density: {density} | "
        f"Occupied Area: {density_percentage:.2f}%"
    )

cap.release()
cv2.destroyAllWindows()

print("\nTASK 10 COMPLETED SUCCESSFULLY!")
