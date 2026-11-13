
import cv2
import os

print("VISIONPORT - TASK 4")
print("Load and Play Traffic Video")

# Specify video path
video_path = "videos/traffic_sample.mp4"

# Check if video exists
if not os.path.exists(video_path):
    print("Error: Video file not found!")
    print("Place traffic_sample.mp4 inside the videos folder.")
    exit()

# Load video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video loaded successfully!")
print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Total Frames:", total_frames)

print("\nPlaying video...")
print("Press Q to stop playback.")

# Play video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video playback completed!")
        break

    cv2.imshow("VisionPort - Traffic Video", frame)

    # Wait according to video FPS
    delay = max(1, int(1000 / fps)) if fps > 0 else 30

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video playback stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("TASK 4 COMPLETED SUCCESSFULLY!")
