
import cv2
import os
import numpy as np

print("=" * 50)
print("VISIONPORT - TASK 14")
print("TRAFFIC SIGNAL STATUS DETECTION")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the traffic signal image captured in Task 2
image_path = os.path.join(
    BASE_DIR, "images", "traffic_signal.jpg"
)

# Read traffic signal image
image = cv2.imread(image_path)

if image is None:
    print("Error: Traffic signal image not found!")
    print("Check images/traffic_signal.jpg")
    exit()

print("Traffic signal image loaded successfully!")

# Convert image from BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define colour ranges in HSV
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])

red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])

yellow_lower = np.array([15, 100, 100])
yellow_upper = np.array([35, 255, 255])

green_lower = np.array([40, 50, 50])
green_upper = np.array([90, 255, 255])

# Create colour masks
red_mask1 = cv2.inRange(
    hsv, red_lower1, red_upper1
)

red_mask2 = cv2.inRange(
    hsv, red_lower2, red_upper2
)

red_mask = cv2.bitwise_or(
    red_mask1, red_mask2
)

yellow_mask = cv2.inRange(
    hsv, yellow_lower, yellow_upper
)

green_mask = cv2.inRange(
    hsv, green_lower, green_upper
)

# Count pixels of each colour
red_pixels = cv2.countNonZero(red_mask)
yellow_pixels = cv2.countNonZero(yellow_mask)
green_pixels = cv2.countNonZero(green_mask)

print("\nTRAFFIC SIGNAL ANALYSIS")
print("-" * 35)

print("Red pixels:", red_pixels)
print("Yellow pixels:", yellow_pixels)
print("Green pixels:", green_pixels)

# Determine dominant signal colour
threshold = 20

if max(red_pixels, yellow_pixels, green_pixels) < threshold:
    signal_status = "SIGNAL NOT DETECTED"

elif red_pixels > yellow_pixels and red_pixels > green_pixels:
    signal_status = "RED SIGNAL - STOP"

elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
    signal_status = "YELLOW SIGNAL - WAIT"

else:
    signal_status = "GREEN SIGNAL - GO"

print("\nTRAFFIC SIGNAL STATUS:", signal_status)

# Display result on image
cv2.putText(
    image,
    signal_status,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2
)

# Display original image and colour masks
cv2.imshow("Traffic Signal Detection", image)
cv2.imshow("Red Mask", red_mask)
cv2.imshow("Yellow Mask", yellow_mask)
cv2.imshow("Green Mask", green_mask)

print("\nPress any key in an image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nTASK 14 COMPLETED SUCCESSFULLY!")
