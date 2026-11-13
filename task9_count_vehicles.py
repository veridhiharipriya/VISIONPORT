
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 9")
print("VEHICLE COUNTING SYSTEM")
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

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Counts
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

# Store IDs of vehicles already counted
counted_ids = set()

# Virtual counting line
line_y = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    if line_y is None:
        line_y = height // 2

    # Track detected vehicles
    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (255, 0, 0),
        2
    )

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            track_id = int(box.id[0])

            if class_id not in vehicle_classes:
                continue

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            vehicle_name = vehicle_classes[class_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw tracking ID and vehicle name
            label = f"ID {track_id}: {vehicle_name}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw vehicle centre
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

            # Count when vehicle crosses the line
            if center_y > line_y and track_id not in counted_ids:

                vehicle_counts[vehicle_name] += 1
                counted_ids.add(track_id)

    # Display vehicle counts
    y_position = 30

    for name, count in vehicle_counts.items():

        cv2.putText(
            frame,
            f"{name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_position += 30

    cv2.imshow("VisionPort - Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL VEHICLE COUNT")
print("-" * 35)

total = 0

for name, count in vehicle_counts.items():
    print(f"{name}: {count}")
    total += count

print("-" * 35)
print("Total Vehicles:", total)

print("\nTASK 9 COMPLETED SUCCESSFULLY!")
