
import cv2
import os
import math
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 15")
print("VEHICLE WAITING TIME ESTIMATION")
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

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print("Traffic video opened successfully!")
print("Video FPS:", fps)

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# Store previous positions and waiting information
vehicle_data = {}

# Movement threshold in pixels
MOVEMENT_THRESHOLD = 5

# Waiting threshold in seconds
WAITING_THRESHOLD = 3

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

    # Detect and track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        tracker="bytetrack.yaml",
        conf=0.40,
        verbose=False
    )

    for result in results:

        if result.boxes.id is None:
            continue

        for box in result.boxes:

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )

            # Calculate vehicle centre
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_position = (center_x, center_y)

            # First appearance of vehicle
            if track_id not in vehicle_data:

                vehicle_data[track_id] = {
                    "position": current_position,
                    "stopped_since": None,
                    "waiting_time": 0
                }

            else:

                previous_position = vehicle_data[track_id]["position"]

                # Calculate movement distance
                distance = math.sqrt(
                    (center_x - previous_position[0]) ** 2 +
                    (center_y - previous_position[1]) ** 2
                )

                if distance < MOVEMENT_THRESHOLD:

                    # Vehicle is nearly stationary
                    if vehicle_data[track_id]["stopped_since"] is None:

                        vehicle_data[track_id]["stopped_since"] = current_time

                    waiting_time = (
                        current_time -
                        vehicle_data[track_id]["stopped_since"]
                    )

                    vehicle_data[track_id]["waiting_time"] = waiting_time

                else:

                    # Vehicle has started moving
                    vehicle_data[track_id]["stopped_since"] = None
                    vehicle_data[track_id]["waiting_time"] = 0

                vehicle_data[track_id]["position"] = current_position

            waiting_time = vehicle_data[track_id]["waiting_time"]

            # Determine vehicle status
            if waiting_time >= WAITING_THRESHOLD:

                status = "WAITING"
                color = (0, 0, 255)

            else:

                status = "MOVING"
                color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display waiting information
            label = f"ID {track_id}: {status}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            time_label = f"Waiting: {waiting_time:.1f} sec"

            cv2.putText(
                frame,
                time_label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "VisionPort - Vehicle Waiting Time",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVEHICLE WAITING TIME SUMMARY")
print("-" * 40)

for track_id, data in vehicle_data.items():

    print(
        f"Vehicle ID {track_id}: "
        f"{data['waiting_time']:.1f} seconds"
    )

print("\nVehicle waiting time estimation completed!")
print("TASK 15 COMPLETED SUCCESSFULLY!")
