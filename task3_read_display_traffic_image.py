
import cv2
import os

print("VISIONPORT - TASK 3")
print("Read and Display Traffic Image")

# Specify image path
image_path = "images/traffic_signal.jpg"

# Check whether image exists
if not os.path.exists(image_path):
    print("Error: Image file not found!")
    print("Please complete Task 2 first.")
    exit()

# Read image using OpenCV
image = cv2.imread(image_path)

# Check whether image was loaded
if image is None:
    print("Error: Unable to read image.")
    exit()

# Display image information
height, width, channels = image.shape

print("Image loaded successfully!")
print("Image Width:", width)
print("Image Height:", height)
print("Number of Channels:", channels)

# Display image
cv2.imshow("VisionPort - Traffic Image", image)

print("Press any key on the image window to close it.")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("TASK 3 COMPLETED SUCCESSFULLY!")
