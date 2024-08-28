# Real-Time Inference Application

This repository contains two Python scripts designed for real-time object detection using a pre-trained YOLOv8 model. The application provides a graphical user interface (GUI) for configuring inference parameters and running the detection script.

## Overview

1. **LiveApp.py**: A Tkinter-based GUI application for configuring parameters and running the inference script (`live_script.py`). It allows users to dynamically set various parameters and execute the script with the specified configurations.

2. **live_script.py**: A real-time inference script that uses YOLOv8 for object detection. It captures video from a specified webcam, performs inference, and allows interactive control of various image properties.

## LiveApp.py

### Features
- **Detect Web Cameras**: Identifies all connected webcams by testing video capture devices with incrementing IDs.
- **GUI Configuration**: Provides fields for adjusting:
  - `Inference Size`: The resolution of input images.
  - `Confidence Threshold`: Minimum confidence score for detections.
  - `IOU Threshold`: Intersection over Union threshold for non-max suppression.
  - `Box Thickness`: Thickness of bounding boxes around detections.
  - `Font Size`: Size of text annotations.
  - `Webcam Path`: ID/path of the webcam.
- **Update and Reset Values**: Functions to update text fields with current settings and reset them to default values.
- **Run Script**: Executes `live_script.py` with the configured parameters using the `subprocess` module.

### Usage
1. Run `LiveApp.py` to open the GUI.
2. Configure the parameters as needed.
3. Click "Run Script" to start inference with the specified settings.

## live_script.py

### Features
- **Real-Time Inference**: Uses YOLOv8 to perform object detection on video captured from a webcam.
- **Interactive Controls**: 
  - Press `q` to exit.
  - Press `i` to toggle inference.
  - Press `m` to toggle the color of bounding boxes.
  - Press `c` and `v` to adjust contrast.
  - Press `b` and `n` to adjust brightness.
- **System Information**: Displays Python, OpenCV, and Ultralytics versions.

### Usage
1. Ensure that your webcam is connected and functioning.
2. Run `live_script.py` with the desired arguments:
   - `--inference_size`: Size of inference (default: 640).
   - `--conf_threshold`: Confidence threshold (default: 0.5).
   - `--iou_threshold`: IoU threshold (default: 0.4).
   - `--box_thickness`: Thickness of bounding boxes (default: 0.5).
   - `--font_size`: Font size for annotations (default: 0.6).
   - `--webcam_path`: ID of the webcam (default: 1).

## Dependencies
- Python 3.x
- OpenCV
- Ultralytics (YOLOv8)
- Tkinter

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/juanjosecas/YOLO-cell-count.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository
   ```
3. Install the required dependencies:
   ```bash
   pip install opencv-python ultralytics
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- YOLOv8 for state-of-the-art object detection.
- OpenCV for video capture and image processing.
- Tkinter for GUI development.
