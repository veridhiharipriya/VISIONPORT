
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 7")
print("MULTIPLE VEHICLE DETECTION USING YOLO")
print("=" * 50)

# Get project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Traffic video path
video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

print("Traffic video opened successfully!")
print("Detecting multiple vehicle types...")

# COCO class IDs
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Detection counters
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Check whether detected object is a vehicle
            if class_id in vehicle_classes and confidence > 0.40:

                vehicle_name = vehicle_classes[class_id]

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )

                # Increase detection count
                vehicle_counts[vehicle_name] += 1

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display vehicle name and confidence
                label = f"{vehicle_name}: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("VisionPort - Multiple Vehicle Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVEHICLE DETECTION SUMMARY")
print("-" * 35)

for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

print("\nTASK 7 COMPLETED SUCCESSFULLY!")
