"""
Webcam Image Capture Tool for ProTac Dataset

Description:
This script captures frames from the webcam and allows the user to save images
interactively by pressing keys.

Key Features:
    - Press 's' to save an image to the dataset folder
    - Press 'q' to quit the application

Output:
    - Saved images are stored in: Dataset/dataset_ProTac_5x5/images/train/
      with filenames: photoNEWERPROTAC_<index>.jpg

Controls:
    - 's' : Save current frame as image
    - 'q' : Quit the program

Dependencies:
    - opencv-python
    - numpy
    - os
    - v4l2-ctl (for Linux camera control)
"""


import cv2
import os
import numpy as np

# Folder setup
img_dir = r'Dataset\dataset_ProTac_5x5\images\train'
os.makedirs(img_dir, exist_ok=True)

#Optional camera tuning
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=50")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=brightness=3")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=contrast=25")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=hue=-2")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=saturation=60")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=sharpness=2")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=gamma=100")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=1")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=gain=0")
# os.system("v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=1") # 1=50Hz, 2=60Hz


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")
img_count = 31 


print("🎥 Press 's' to capture image, 'q' to quit.")

last_image = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("👋 Exiting.")
        break

    elif key == ord('s'):
        img_path = os.path.join(img_dir, f"photoNEWERPROTAC_{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        last_image = frame.copy()
        print(f"📸 Saved {img_path}")
        img_count += 1
   

cap.release()
cv2.destroyAllWindows()
