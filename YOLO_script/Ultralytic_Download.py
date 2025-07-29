"""
YOLO Marker Detection Download Training Script

Description:
This script loads a specific YOLO model from Ultralytics Hub and begins training
with the dataset specified in Ultralytics settings or configuration file. The 
script supports multiple model checkpoints for tracking different marker types.

Available Models (comment/uncomment as needed):
    - YOLO11s (generic marker)
    - YOLO11s_6x6marker
    - YOLO8s_6x6marker
    - YOLO11s_5x5marker
    - YOLOv8s_5x5marker (current active)

Output:
    - Training results saved in the `runs/` directory by Ultralytics

Dependencies:
    - ultralytics

"""


from ultralytics import YOLO, checks, hub
checks()

#YOLO11s
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
# model = YOLO('https://hub.ultralytics.com/models/UU3rVGl537K2rUK0mVuV')

#YOLO11s_6x6marker
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
# model = YOLO('https://hub.ultralytics.com/models/7pzulqXUOSNzLhFJ7bed')


#YOLO8s_6x6marker
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
# model = YOLO('https://hub.ultralytics.com/models/8T0sGy75jlKSG7Y8qH3I')


#YOLO11s_5x5Marker
# hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
# model = YOLO('https://hub.ultralytics.com/models/AUpSwxtN8Xcew47O87nj')

#YOLO8s_5x5Marker
hub.login('15fe7644b64af0034a4aa59afdbcc14d630c9922fc')
model = YOLO('https://hub.ultralytics.com/models/Rr0ta5z4hkQFFLPyMKbM')


results = model.train()

#results = model.predict(source=0,show=True)

