import warnings as wrn
wrn.filterwarnings("ignore")

print("Loading Ultralytics...\n\n")
import ultralytics
from ultralytics import YOLO

import cv2
import numpy as np
import psutil
import time as tm
import os as os
import sys as sys
import argparse
import platform as plt
import glob

# Welcome message with instructions and system info
welcome_message = f"""
Welcome to real-time inference with YOLOv8.

This script allows real-time inference using a pre-trained YOLOv8 model.

Keys:
To exit the script, press 'q'.
To toggle inference, press 'i'.
To toggle the color of the boxes, press 'm'.
To increase contrast, press 'c'.
To decrease contrast, press 'v'.
To increase brightness, press 'b'.
To decrease brightness, press 'n'.

You are using Python {sys.version} on {plt.system()}.
The OpenCV version is {cv2.__version__}.
The Ultralytics version is {ultralytics.__version__}.
"""

print(welcome_message)

# Argument parser configuration
parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('--inference_size', type=int, default=640, help='Size of inference')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--iou_threshold', type=float, default=0.4, help='IoU (Intersection over Union) threshold')
parser.add_argument('--box_thickness', type=float, default=0.5, help='Thickness of the bounding box')
parser.add_argument('--font_size', type=float, default=0.6, help='Font size for text annotations')
parser.add_argument('--webcam_path', type=int, default=1, help='Webcam path')

# Parse the arguments
args = parser.parse_args()

# Control variable for inference
device = 'cpu'  # 'cpu', 0, 1, 2, etc. to select the device
local_model_path = 'RCCD-8n640.onnx'  # Path to the ONNX model

# Assign argument values to variables
conf = args.conf_threshold
iou = args.iou_threshold
inference_size = args.inference_size
webcam_path = args.webcam_path  # '0' for the default webcam
font_size = args.font_size
box_thickness = args.box_thickness

def detect_web_cameras():
    # Initialize the OpenCV video capture to search for connected web cameras
    cam_ids = []  # List to store detected camera IDs
    index = 0  # Start testing from ID 0

    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Attempt to open the camera with the current ID
        
        if not cap.isOpened():
            break  # If the camera cannot be opened, exit the loop
        
        ret, frame = cap.read()  # Attempt to read a frame from the camera
        
        if ret:
            cam_ids.append(index)  # If a frame is read successfully, add the camera ID to the list
        
        cap.release()  # Release the camera resource
        index += 1  # Test the next ID
    
    return cam_ids  # Return the list of detected camera IDs

detected_cameras = detect_web_cameras()
print("Detected web cameras:", detected_cameras)

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(webcam_path)

#if not cap.isOpened():
#    print("Error: Cannot open the camera")
#    exit()

# Variable to control inference
perform_inference = False
# Initialize contrast and brightness values
contrast = 1.0
brightness = 0

# Define the desired resolution for inference
desired_width = 1024
desired_height = 768

print('\n\n ====== Modules loaded successfully =====')

# Class to temporarily suppress console output
class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr

# Function to generate a code based on the current date and time to use as part of the screenshot filename
def get_time_code():
    return tm.strftime("%Y%m%d_%H%M%S", tm.localtime())

# Load the pre-trained YOLOv8 model
model = YOLO(local_model_path)

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break

    # Resize the frame to the desired resolution
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Adjust contrast and brightness
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    if perform_inference:
        # Use the suppress_output class to temporarily suppress console output during inference
        with suppress_output(suppress_stdout=True, suppress_stderr=False):
            results = model(frame, task='predict', device=device, imgsz=inference_size, conf=conf, iou=iou)

        # Perform inference on the frame
        # results = model.predict(frame, task='predict', imgsz=inference_size, conf=0.30, iou=0.45)

        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Generate random colors in RGB for the boxes
                if multicolor_box:
                    red_colors = np.random.randint(32, 228)
                    green_colors = np.random.randint(32, 228)
                    blue_colors = np.random.randint(32, 228)
                else:
                    red_colors = 255
                    green_colors = 128
                    blue_colors = 32
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (red_colors, green_colors, blue_colors), box_thickness)

            # Uncomment to show labels and confidences
            # for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            #     label = f"{model.names[int(cls)]}: {conf:.2f}"
            #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            num_total_objects = len(result.boxes.xyxy)
            cv2.putText(frame, f'Cells: {num_total_objects}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 128, 0), 2)

    # Indicate the key to exit
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    cv2.putText(frame, f'CPU: {cpu_usage}%, RAM: {ram_usage}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, (128, 255, 32), 1)
    help_message1 = "'q' to exit, 'i' Inference, 'm' Multicolor, 'c' (+)contrast, 'v' (-)contrast"
    help_message2 = "'b' (+)brightness, 'n' (-)brightness, 'p' Screenshot"
    cv2.putText(frame, help_message1, (10, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, font_size, (128, 255, 0), 1)
    cv2.putText(frame, help_message2, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (128, 255, 0), 1)

    # Show the frame
    cv2.imshow('Frame', frame)
    
    # Wait for key press to exit ('q') or toggle inference ('i')
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        perform_inference = not perform_inference
    elif key == ord('m'):
        multicolor_box = not multicolor_box
    elif key == ord('c'):
        contrast += 0.1  # Increase contrast
    elif key == ord('v'):
        contrast -= 0.1  # Decrease contrast
    elif key == ord('b'):
        brightness += 10  # Increase brightness
    elif key == ord('n'):
        brightness -= 10  # Decrease brightness
    elif key == ord('p'):  # To take a screenshot of the current frame
        cv2.imwrite(f'Exp_{get_time_code()}.png', frame)
        print(f"Screenshot saved as Exp_{get_time_code()}.png")

# Release and close
cap.release()
cv2.destroyAllWindows()
print(" ======= BYE BYE! =======")
