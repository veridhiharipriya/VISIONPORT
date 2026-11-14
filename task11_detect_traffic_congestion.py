
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 11")
print("TRAFFIC CONGESTION DETECTION")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

print("Traffic video opened successfully!")

vehicle_classes = [2, 3, 5, 7]

# Congestion thresholds based on detected vehicle count
LOW_LIMIT = 5
MEDIUM_LIMIT = 12

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect vehicles
    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.40,
        verbose=False
    )

    vehicle_count = 0

    for result in results:

        for box in result.boxes:

            vehicle_count += 1

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            # Draw bounding boxes
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # Determine congestion level
    if vehicle_count <= LOW_LIMIT:
        congestion = "LOW - NORMAL TRAFFIC"
        display_color = (0, 255, 0)

    elif vehicle_count <= MEDIUM_LIMIT:
        congestion = "MEDIUM - MODERATE TRAFFIC"
        display_color = (0, 255, 255)

    else:
        congestion = "HIGH - TRAFFIC CONGESTION"
        display_color = (0, 0, 255)

    # Display results
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
        congestion,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        display_color,
        2
    )

    cv2.imshow(
        "VisionPort - Traffic Congestion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nTraffic congestion detection completed!")
print("TASK 11 COMPLETED SUCCESSFULLY!")
