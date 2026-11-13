
import cv2
import os
from ultralytics import YOLO

print("=" * 50)
print("VISIONPORT - TASK 6")
print("CAR DETECTION USING YOLO")
print("=" * 50)

# Get the project folder location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Locate the traffic video
video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

# Load the pre-trained YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

# Open the traffic video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

print("\nTraffic video opened successfully!")
print("Detecting cars...")

car_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame, verbose=False)

    # Draw bounding boxes around cars
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO class ID 2 represents a car
            if class_id == 2 and confidence > 0.40:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )

                car_count += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"Car: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("VisionPort - Car Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nDetection completed!")
print("Total car detections across processed frames:", car_count)
print("\nTASK 6 COMPLETED SUCCESSFULLY!")
