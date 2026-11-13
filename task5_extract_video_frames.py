
import cv2
import os

print("=" * 50)
print("VISIONPORT - TASK 5")
print("Extract Individual Frames from Traffic Video")
print("=" * 50)

# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input video path
video_path = os.path.join(
    BASE_DIR, "videos", "traffic_sample.mp4"
)

# Output folder
output_folder = os.path.join(
    BASE_DIR, "extracted_frames"
)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open traffic video!")
    exit()

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully!")
print("FPS:", fps)
print("Total Frames:", total_frames)

# Extract one frame every 30 frames
frame_interval = 30

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:

        filename = f"frame_{saved_count + 1:04d}.jpg"

        output_path = os.path.join(
            output_folder, filename
        )

        cv2.imwrite(output_path, frame)

        saved_count += 1

        print("Saved:", filename)

    frame_count += 1

# Release video
cap.release()

print("\nExtraction completed!")
print("Total frames processed:", frame_count)
print("Total images saved:", saved_count)
print("Saved location:", output_folder)

print("\nTASK 5 COMPLETED SUCCESSFULLY!")
