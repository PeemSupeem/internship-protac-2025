"""
Real-Time Marker Detection and Tracking using YOLO and SORT

Description:
This script performs real-time detection and tracking of 5x5 ProTac markers from webcam input.
It uses:
    - YOLOv8 (Ultralytics) segmentation model to detect marker regions
    - Contour extraction and ellipse fitting for shape representation
    - SORT tracker to assign and maintain consistent object IDs across frames
    - Track memory to render fake ellipses for recently missing markers

Outputs:
    - Tracked bounding boxes with consistent ID overlays
    - Ellipses drawn for matched contours and faded fake markers

Controls:
    - 'q': Quit the program

Dependencies:
    - opencv-python
    - numpy
    - ultralytics
    - sort.py (local SORT implementation)
"""

import cv2
import sys
import numpy as np
from ultralytics import YOLO, checks, hub
from collections import deque


from sort import Sort

# Initialization
checks()

#YOLO8s_5x5Marker
hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
model = YOLO('https://hub.ultralytics.com/models/Rr0ta5z4hkQFFLPyMKbM')

cap = cv2.VideoCapture(0)
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

track_memory = {}  # Store last known ellipses
MAX_MISS_FRAMES = 10


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False, imgsz=960)
    mask_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    dets = []

    for r in results:
        if r.masks is not None:
            for m in r.masks.xy:
                pts = np.array(m, dtype=np.int32)
                cv2.fillPoly(mask_canvas, [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_canvas = np.zeros_like(frame)

    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            score = 0.7
            dets.append([x, y, x + w, y + h, score])

    if len(dets) > 0:
        dets = np.array(dets)
        tracks = tracker.update(dets)
        active_ids = set()

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            active_ids.add(track_id)


            # Draw rectangle 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Match contour to track
            matched_ellipse = None
            for cnt in contours:
                if len(cnt) >= 5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        matched_ellipse = cv2.fitEllipse(cnt)
                        break

            if matched_ellipse:
                track_memory[track_id] = {'ellipse': matched_ellipse, 'missed': 0}

        # Update memory for missing tracks
        for tid in list(track_memory.keys()):
            if tid not in active_ids:
                track_memory[tid]['missed'] += 1
                if track_memory[tid]['missed'] > MAX_MISS_FRAMES:
                    del track_memory[tid]

    # Draw ellipses from memory
    for tid, data in track_memory.items():
        ellipse = data['ellipse']
        if data['missed'] == 0:
            color = (0, 255, 0)
        else:
            fade = 255 - data['missed'] * int(255 / MAX_MISS_FRAMES)
            color = (fade, fade, fade)
        cv2.ellipse(ellipse_canvas, ellipse, color, -1)
    

    cv2.imshow("Tracked Frame", frame)
    cv2.imshow("Ellipses Canvas", ellipse_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
