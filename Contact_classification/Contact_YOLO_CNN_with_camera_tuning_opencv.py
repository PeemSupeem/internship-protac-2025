

"""
Contact Direction Classification System using YOLO and CNN

Description:
This script performs real-time classification of contact direction based on webcam
footage. It uses:
    - YOLO (Ultralytics) for bounding box/segmentation-based region detection
    - Contour + ellipse fitting to isolate object shape
    - Pretrained CNN model (PyTorch .pth) to classify contact 

Outputs classification label:
    [ "Contact", "NonContact" ]

Also prints inference time and estimated FPS.

Controls:
    - 'q': Quit the program

Dependencies:
    - opencv-python
    - numpy
    - ultralytics
    - PIL
    - torch
    - torchvision
    
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO, hub
import time 

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
model = YOLO('YOLO_model\yolov8n.pt')
model = YOLO('YOLO_model\yolov8s-seg.pt')
model = YOLO('YOLO_model\yolo11s-seg.pt')

# Load trained CNN model
cnn_model = torch.load('Contact_classifier_model\contact_classifier_region\contact_classifier_region2.pth', weights_only=False)
cnn_model.eval()

# CNN transform (match training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_CONTRAST, 34)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_SATURATION, 120)
cap.set(cv2.CAP_PROP_SHARPNESS, 2)             
cap.set(cv2.CAP_PROP_GAMMA, 100)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FPS, 50)    

while True:

    start_time = time.time()  # Start timing

    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, conf=0.3, verbose=False, imgsz=640)

    mask_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipse_canvas = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 80:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(ellipse_canvas, ellipse, (255, 255, 255), -1)

    pil_img = Image.fromarray(ellipse_canvas)
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = "Contact" if predicted.item() == 0 else "Non-Contact"

    elapsed_time = time.time() - start_time  # End timing
    fps = 1.0 / elapsed_time
    print(f"Prediction: {label} | Time: {elapsed_time:.3f} seconds | FPS: {fps:.2f}")

    cv2.imshow("Binary Marker Canvas", ellipse_canvas)

    cv2.imshow("Realframe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
