
"""
Title: Real-time Marker Detection with YOLO + SORT Tracker


Description:
This script uses a YOLO segmentation model (Ultralytics) and 
SORT tracker to detect and track elliptical contact markers 
from webcam input, apply a void mask, and save frames based 
on keyboard interaction.

Keyboard Controls:
    - 'c': Switch to CONTACT save mode
    - 'n': Switch to NON-CONTACT save mode
    - 's': Save current frame to selected class folder
    - 'q': Quit the program

Dependencies:
    - opencv-python
    - numpy
    - ultralytics
    - sort.py (tracking)
    - Pretrained YOLOv8 model (.pt)
    - Void mask image
"""



import cv2
import numpy as np
import os
import time
from ultralytics import YOLO, hub

import sys
from sort import Sort

# Setup save directories
os.makedirs(r'Contact_classification_dataset\class2\non_contact', exist_ok=True)
os.makedirs(r'Contact_classification_dataset\class2\non_contact', exist_ok=True)

# Load YOLO model from Ultralytic
#YOLO11sOldmarker
#model = YOLO('https://hub.ultralytics.com/models/7pzulqXUOSNzLhFJ7bed')
#YOLO11sNewmarker
#model = YOLO('https://hub.ultralytics.com/models/AUpSwxtN8Xcew47O87nj')
#YOLOv8m
#model = YOLO('https://hub.ultralytics.com/models/kHnez0KPoAKzkdMsJJTV')  
#YOLOv8n
model = YOLO('https://hub.ultralytics.com/models/QAcIeHrBQxyD9IW4RYqk')

# Load YOLO model from Relative path
model = YOLO('YOLO_model\yolov8n.pt')
model = YOLO('YOLO_model\yolov8s-seg.pt')
model = YOLO('YOLO_model\yolo11s-seg.pt')

# Load the void mask image
void_mask = cv2.imread('VoidOverlay\TheVoid2.png', cv2.IMREAD_GRAYSCALE)

# Initialize SORT tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
track_memory = {}
MAX_MISS_FRAMES = 10

# Capture from webcam
cap = cv2.VideoCapture(0)
save_mode = None  # 'contact' or 'non_contact'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize void mask to match frame
    void_resized = cv2.resize(void_mask, (frame.shape[1], frame.shape[0]))
    void_resized = cv2.bitwise_not(void_resized)

    # Apply void mask
    frame_masked = cv2.bitwise_and(frame, frame, mask=void_resized)

    # Run YOLO
    results = model.predict(source=frame_masked, conf=0.2, verbose=False)

    # Canvas for mask
    mask_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    dets = []

    for r in results:
        if r.masks is not None:
            for m in r.masks.xy:
                pts = np.array(m, dtype=np.int32)
                cv2.fillPoly(mask_canvas, [pts], 255)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Contours and detections
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            score = 0.7
            dets.append([x, y, x + w, y + h, score])

    # Tracking
    canvas = np.zeros_like(frame)
    if len(dets) > 0:
        dets = np.array(dets)
        tracks = tracker.update(dets)
        active_ids = set()

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            active_ids.add(track_id)

            matched_ellipse = None
            for cnt in contours:
                if len(cnt) >= 5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        matched_ellipse = cv2.fitEllipse(cnt)
                        break

            if matched_ellipse:
                track_memory[track_id] = {'ellipse': matched_ellipse, 'missed': 0}

        for tid in list(track_memory.keys()):
            if tid not in active_ids:
                track_memory[tid]['missed'] += 1
                if track_memory[tid]['missed'] > MAX_MISS_FRAMES:
                    del track_memory[tid]

    # Draw ghost markers
    for tid, data in track_memory.items():
        ellipse = data['ellipse']
        if data['missed'] == 0:
            color = (255, 255, 255)
        else:
            fade = 255 - data['missed'] * int(255 / MAX_MISS_FRAMES)
            color = (fade, fade, fade)
        cv2.ellipse(canvas, ellipse, color, -1)

    # Convert to 3-channel for saving
    binary_rgb = canvas

    # Display
    #cv2.putText(binary_rgb, f"{save_mode or 'None'}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Binary Marker Canvas", binary_rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        save_mode = 'contact'
        print("Switched to CONTACT save mode")
    elif key == ord('n'):
        save_mode = 'non_contact'
        print("Switched to NON-CONTACT save mode")
    elif key == ord('s') and save_mode:
        filename = f"Contact_classification_dataset\class2{save_mode}\frame_{int(time.time())}.jpg"
        cv2.imwrite(filename, binary_rgb)
        print(f"Saved frame to {save_mode} folder")

cap.release()
cv2.destroyAllWindows()
