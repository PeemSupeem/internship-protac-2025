"""
Real-Time Marker Tracking System using YOLO and SORT

Description:
This script detects and tracks elliptical markers from webcam input in real time.
It uses:
    - YOLO (Ultralytics) segmentation model for object detection
    - Contour extraction and ellipse fitting for shape representation
    - SORT algorithm to assign and maintain tracking IDs

Outputs:
    - Bounding box with tracking ID
    - Fitted ellipse for each detected marker
    - ID overlay on live video

Controls:
    - 'q': Quit the program

Dependencies:
    - opencv-python
    - numpy
    - ultralytics
    - sort.py (local tracker file)
"""


import cv2
import sys
import numpy as np
from ultralytics import YOLO, checks, hub
import random

from sort import Sort


# YOLO11s model
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
# model = YOLO('https://hub.ultralytics.com/models/UU3rVGl537K2rUK0mVuV')


# Load YOLO model from Ultralytic
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
#YOLO11sOldmarker
#model = YOLO('https://hub.ultralytics.com/models/7pzulqXUOSNzLhFJ7bed')
#YOLO11sNewmarker
#model = YOLO('https://hub.ultralytics.com/models/AUpSwxtN8Xcew47O87nj')
#YOLOv8m
#model = YOLO('https://hub.ultralytics.com/models/kHnez0KPoAKzkdMsJJTV')  
#YOLOv8n
#model = YOLO('https://hub.ultralytics.com/models/QAcIeHrBQxyD9IW4RYqk')

# Load YOLO model from Relative path
# model = YOLO('YOLO_model\yolov8n.pt')
# model = YOLO('YOLO_model\yolov8s-seg.pt')
model = YOLO('YOLO_model\yolo11s-seg.pt') # ← Currently active


tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)



# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with YOLO segmentation
    results = model.predict(source=frame, conf=0.3, verbose=False, imgsz=960)

    mask_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    dets = []

    for r in results:
        if r.masks is not None:
            for m in r.masks.xy:
                pts = np.array(m, dtype=np.int32)
                cv2.fillPoly(mask_canvas, [pts], 255)

    # Clean mask for better contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Ellipse and detection boxes
    ellipse_canvas = np.zeros_like(frame)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 80:
            ellipse = cv2.fitEllipse(cnt)

            
            
            cv2.ellipse(ellipse_canvas, ellipse, (255, 255, 255), -1)

        x, y, w, h = cv2.boundingRect(cnt)
        score = 0.7  # fixed confidence
        dets.append([x, y, x + w, y + h, score])

    # Update tracker
    if len(dets) > 0:
        dets = np.array(dets)
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
      
    else:
        print("No detections passed to tracker.")

    # Show outputs
    cv2.imshow("Tracked Frame", frame)
    cv2.imshow("Ellipses Canvas", ellipse_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
