import tkinter as tk
from tkinter import ttk
import subprocess
import cv2

"""
This script creates a graphical user interface (GUI) using Tkinter for configuring parameters and running a Python script (`live_script.py`) for real-time inference using a webcam. 

Main functionalities include:

1. **Detect Web Cameras**:
   - Uses OpenCV to detect all connected web cameras by attempting to open video capture devices with incrementing IDs until no more cameras are found. Detected camera IDs are printed to the console.

2. **GUI Configuration**:
   - The GUI allows users to input and adjust several parameters required for the inference script:
     - `Inference Size`: The size of the input images for inference.
     - `Confidence Threshold`: The minimum confidence score for detections.
     - `IOU Threshold`: The Intersection over Union threshold for non-max suppression.
     - `Box Thickness`: The thickness of the bounding boxes drawn around detections.
     - `Font Size`: The size of the font used for text annotations.
     - `Webcam Path`: The ID/path of the webcam to be used.

3. **Updating and Resetting Values**:
   - Functions are provided to update the text fields in the GUI based on current variable values (`update_controls_values`) and to reset the fields to their default values (`reset_default_values`).

4. **Running the Script**:
   - When the user clicks the button to run the script, the GUI collects the current values of the parameters and constructs a command to run `live_script.py` with these values as arguments using the `subprocess` module. This allows for dynamic configuration of the script's behavior based on user input in the GUI.

The GUI is designed for ease of use, allowing users to configure and run the script without manually editing code, thus streamlining the process of testing and deploying different parameter settings.
"""


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

# Function to update the values of the text fields in the GUI
def update_controls_values():
    inference_size_entry.delete(0, tk.END)
    inference_size_entry.insert(0, str(inference_size_var.get()))

    conf_entry.delete(0, tk.END)
    conf_entry.insert(0, str(conf_var.get()))

    iou_entry.delete(0, tk.END)
    iou_entry.insert(0, str(iou_var.get()))

    box_thickness_entry.delete(0, tk.END)
    box_thickness_entry.insert(0, str(box_thickness_var.get()))

    font_size_entry.delete(0, tk.END)
    font_size_entry.insert(0, str(font_size_var.get()))

    webcam_path_entry.delete(0, tk.END)
    webcam_path_entry.insert(0, str(webcam_path_var.get()))

# Function to reset the default values in the GUI
def reset_default_values():
    inference_size_var.set("640")  # Default inference size
    conf_var.set("0.5")  # Default confidence threshold
    iou_var.set("0.4")  # Default IOU (Intersection over Union) threshold
    box_thickness_var.set("0.5")  # Default box thickness
    font_size_var.set("0.6")  # Default font size
    webcam_path_var.set("0")  # Default webcam path

# Function to run the script with the options configured from the GUI
def run_script():
    # Get the current values of the variables from the GUI
    inference_size = inference_size_var.get()
    conf_threshold = conf_var.get()
    iou_threshold = iou_var.get()
    box_thickness = box_thickness_var.get()
    font_size = font_size_var.get()
    webcam_path = webcam_path_var.get()

    # Build the command to call live_script.py with the configured arguments
    command = ['python', 'live_script.py',
               '--inference_size', inference_size,
               '--conf_threshold', conf_threshold,
               '--iou_threshold', iou_threshold,
               '--box_thickness', box_thickness,
               '--font_size', font_size,
               '--webcam_path', webcam_path]

    # Execute the command in a subprocess
    subprocess.run(command)

    # Confirmation message
    print("Script executed with the options configured from the GUI.")

# Create the main window of the GUI
root = tk.Tk()
root.title("Parameter Configuration")

# Configuration variables
inference_size_var = tk.StringVar()
conf_var = tk.StringVar()
iou_var = tk.StringVar()
box_thickness_var = tk.StringVar()
font_size_var = tk.StringVar()
webcam_path_var = tk.StringVar()

# Initial configuration values
inference_size_var.set("640")
conf_var.set("0.5")
iou_var.set("0.4")
box_thickness_var.set("0.5")
font_size_var.set("0.6")
webcam_path_var.set("0")

# Labels and entry fields for each configuration variable
ttk.Label(root, text="Inference Size:").pack()
inference_size_entry = ttk.Entry(root, textvariable=inference_size_var)
inference_size_entry.pack()

ttk.Label(root, text="Confidence Threshold:").pack()
conf_entry = ttk.Entry(root, textvariable=conf_var)
conf_entry.pack()

ttk.Label(root, text="IOU Threshold:").pack()
iou_entry = ttk.Entry(root, textvariable=iou_var)
iou_entry.pack()

ttk.Label(root, text="Box Thickness:").pack()
box_thickness_entry = ttk.Entry(root, textvariable=box_thickness_var)
box_thickness_entry.pack()

ttk.Label(root, text="Font Size:").pack()
font_size_entry = ttk.Entry(root, textvariable=font_size_var)
font_size_entry.pack()

ttk.Label(root, text="Webcam Path:").pack()
webcam_path_entry = ttk.Entry(root, textvariable=webcam_path_var)
webcam_path_entry.pack()

# Buttons to reset default values and run the script
ttk.Button(root, text="Reset Default Values", command=reset_default_values).pack()
ttk.Button(root, text="Run Script", command=run_script).pack()

# Main function of the GUI
root.mainloop()
