# ProTac Real-Time Contact Classification and Marker Detection
This repository contains real-time vision-based pipelines for contact marker detection, classification, dataset generation, and camera tuning using YOLO, CNN, and OpenCV. Scripts are organized for both training and deployment on ProTac-integrated robotic systems. This repository includes several experimental scripts involving YOLO-based methods. Some scripts were developed through trial and error and may not be fully functional or optimized.

# Binary mask dataset preparation

## Real-time Marker Detection with YOLO + SORT Tracker
This script uses a YOLO segmentation model (Ultralytics) and SORT tracker to detect and track elliptical contact markers from webcam input, apply a void mask, and save frames based on keyboard interaction.
```
YOLOBinaryDataset.py
```

## Real-Time Directional Contact Classification using YOLOv8 + SORT Tracker
This script captures live video from the webcam, applies a void mask to ignore background areas, runs a YOLO segmentation model to detect contact markers, and uses the SORT tracker to track and match them over time. Ellipses are fitted to detected contours, and users can save masked output frames into direction-specific folders for later classification training.
```
YOLOBinaryRegion.py
```

# Camera

## Webcam Manual Tuning using v4l2-ctl
This script configures a Linux webcam using `v4l2-ctl` commands to manually set exposure and image properties for more consistent video capture.
```
camera_tuning.py
```
## Webcam Image Capture Tool for ProTac Dataset
This script captures frames from the webcam and allows the user to save images interactively by pressing keys. The images can be used for annotation ()
```
takingphoto.py
```

## Dataset Annotation

All training images were annotated using [CVAT](https://app.cvat.ai/tasks) with YOLO segmentation format, primarily using polygon-shaped annotations for masking.


# Contact classification

## Contact Classification with YOLO Box Detection and Ellipse Filtering
This script performs real-time classification of contact state using YOLO with class labels
```
Contact_YOLO_CNN_BlackandWhite_classes.py
```
## Contact Detection System using YOLO and CNN Classifier with Camera Tuning (v4l2-ctl for linux)
This script performs real-time classification of contact state using YOLO with camera tuning
```
Contact_YOLO_CNN_with_camera_tuning_v4l2_ctl.py
```
## Contact Detection System using YOLO and CNN Classifier with Camera Tuning (opencv) 
This script performs real-time classification of contact state using YOLO with camera tuning
```
Contact_YOLO_CNN_with_camera_tuning_opencv.py
```
## Contact Direction Classification System using YOLO and CNN
This script performs real-time classification of contact direction based on webcam
```
Simple_Contact_region.py
```
## Contact Direction Classification System using YOLO and CNN (region reduced)
This script performs real-time classification of contact direction based on webcam
```
Simple_Contact_region_5x5markers.py
```

# YOLO Model Export to ONNX Format
This script loads a YOLO PyTorch model and exports it to ONNX format
```
UltralytictoONNX.py
```

# YOLO Script


## Real-Time Marker Tracking System using YOLO and SORT
This script detects and tracks elliptical markers from webcam input in real time.
```
Segmentation_ID.py
```
## YOLO Marker Detection Download Training Script
This script loads a specific YOLO model from Ultralytics Hub and begins training with the dataset specified in Ultralytics settings or configuration file using [Ultralytics HUB](https://hub.ultralytics.com/).  

```
Ultralytic_Download.py
```

## Dynamic Contact Classification with YOLO, CNN, and ROI Switching
This script performs real-time contact detection from webcam using YOLO with ROI fallback mode when global detection fails.
```
YOLO_ROI_mode.py
```
## Real-Time Marker Detection and Tracking using YOLO and SORT with fake marker
This script performs real-time detection and tracking of 5x5 ProTac markers from webcam input. SORT tracker to assign and maintain consistent object IDs across and track memory to render fake ellipses for recently missing markers

```
Yolo_Sort_with_fake_marker.py
```


## Marker Segmentation with CLAHE and Contour Extraction using YOLO
This script captures webcam input and performs marker segmentation using normal imageprocessing pipeline
```
Yolo_with_Opencv_Normal_processing_pipeline.py
```

# ONNX-based YOLOv5 Inference with Ellipse Mask Generation
This script runs real-time object detection using a YOLOv5 model exported to ONNX.
```
YOLOv5Binary.py
```



