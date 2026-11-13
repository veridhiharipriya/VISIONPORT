
import cv2
import os

print("VISIONPORT - TASK 2")
print("Traffic Signal Image Capture")

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("Camera opened successfully!")
print("Press SPACE to capture an image.")
print("Press ESC to exit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture camera frame.")
        break

    cv2.imshow("VisionPort - Traffic Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE key
        os.makedirs("images", exist_ok=True)

        image_path = "images/traffic_signal.jpg"
        cv2.imwrite(image_path, frame)

        print("Image captured successfully!")
        print("Saved at:", image_path)
        break

    elif key == 27:  # ESC key
        print("Capture cancelled.")
        break

camera.release()
cv2.destroyAllWindows()
