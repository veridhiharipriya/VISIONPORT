
import cv2
import os
import math
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 13")
print("VEHICLE SPEED ESTIMATION")
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

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print("Traffic video opened successfully!")
print("Video FPS:", fps)

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Store previous vehicle positions
previous_positions = {}

# Approximate pixel-to-meter conversion.
# This is only a demonstration value.
PIXELS_PER_METER = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

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

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_position = (center_x, center_y)

            speed_kmh = 0.0

            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                # Calculate pixel distance travelled
                distance_pixels = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                # Convert pixels to approximate meters
                distance_meters = (
                    distance_pixels / PIXELS_PER_METER
                )

                # Calculate speed
                speed_ms = distance_meters * fps

                speed_kmh = speed_ms * 3.6

            previous_positions[track_id] = current_position

            vehicle_name = vehicle_classes[class_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display speed
            label = f"ID {track_id} | {speed_kmh:.1f} km/h"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.circle(
                frame,
                current_position,
                4,
                (0, 0, 255),
                -1
            )

    cv2.imshow(
        "VisionPort - Vehicle Speed Estimation",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVehicle speed estimation completed!")
print("TASK 13 COMPLETED SUCCESSFULLY!")
