
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 12")
print("INDIVIDUAL VEHICLE TRACKING")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

# Open traffic video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

print("Traffic video opened successfully!")
print("Tracking individual vehicles...")

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Track vehicles using ByteTrack
    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        tracker="bytetrack.yaml",
        conf=0.40,
        verbose=False
    )

    for result in results:

        if result.boxes.id is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            if class_id not in vehicle_classes:
                continue

            vehicle_name = vehicle_classes[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display tracking ID and vehicle type
            label = f"ID: {track_id} | {vehicle_name}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Mark vehicle centre
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

    cv2.imshow(
        "VisionPort - Individual Vehicle Tracking",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nIndividual vehicle tracking completed!")
print("TASK 12 COMPLETED SUCCESSFULLY!")
