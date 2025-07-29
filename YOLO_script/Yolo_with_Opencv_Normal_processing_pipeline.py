"""
Marker Segmentation with CLAHE and Contour Extraction using YOLO

Description:
This script captures webcam input and performs marker segmentation using:
    - YOLO (Ultralytics) for bounding box detection
    - Grayscale + CLAHE (adaptive histogram equalization) for contrast enhancement
    - Gaussian blur and binary thresholding
    - Morphological operations to clean noise
    - Contour extraction for feature segmentation within the detected region

Outputs:
    - Bounding boxes from YOLO
    - Processed contours drawn inside each YOLO-detected marker region

Controls:
    - 'q': Quit the program

Dependencies:
    - opencv-python
    - numpy
    - ultralytics

"""


import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics import YOLO, checks, hub

#YOLO8s_5x5Marker
hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
model = YOLO('https://hub.ultralytics.com/models/Rr0ta5z4hkQFFLPyMKbM')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Predict
    results = model.predict(source=frame, verbose=False)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]

            #Convert to grayscale 
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
            gray_eq = clahe.apply(gray)

            blur = cv2.GaussianBlur(gray_eq, (7, 7), 0)

            #Apply adaptive threshold
            _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


            #Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #Draw contours on cropped image
            cv2.drawContours(cropped, contours, -1, (0, 255, 0), 1)

            #Display segmented crop
            #cv2.imshow("Cropped Marker", cropped)
            #cv2.imshow("Thresholded", thresh)

    #Show the frame with boxes
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()