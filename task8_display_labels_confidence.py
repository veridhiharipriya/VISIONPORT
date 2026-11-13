
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 8")
print("DISPLAY VEHICLE LABELS AND CONFIDENCE")
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
print("Displaying vehicle labels and confidence scores...")

# Vehicle class IDs
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

    # Perform YOLO detection
    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id in vehicle_classes and confidence > 0.40:

                vehicle_name = vehicle_classes[class_id]

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )

                # Convert confidence into percentage
                confidence_percentage = confidence * 100

                # Prepare label
                label = (
                    f"{vehicle_name} "
                    f"{confidence_percentage:.1f}%"
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - 30),
                    (x1 + 180, y1),
                    (0, 255, 0),
                    -1
                )

                # Display label
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

    # Display video
    cv2.imshow(
        "VisionPort - Vehicle Labels and Confidence",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVehicle labels and confidence displayed successfully!")
print("\nTASK 8 COMPLETED SUCCESSFULLY!")
