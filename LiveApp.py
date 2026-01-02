import tkinter as tk
from tkinter import ttk, messagebox
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

# Function to validate numeric input
def validate_float(value, min_val=0.0, max_val=10.0):
    """Valida que el valor sea un número flotante dentro del rango especificado"""
    try:
        num = float(value)
        return min_val <= num <= max_val
    except ValueError:
        return False

def validate_int(value, min_val=0, max_val=10000):
    """Valida que el valor sea un entero dentro del rango especificado"""
    try:
        num = int(value)
        return min_val <= num <= max_val
    except ValueError:
        return False

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

    # Validate inputs before running
    if not validate_int(inference_size, 320, 1280):
        messagebox.showerror("Error de Validación", "Inference Size debe ser un entero entre 320 y 1280")
        return
    
    if not validate_float(conf_threshold, 0.0, 1.0):
        messagebox.showerror("Error de Validación", "Confidence Threshold debe ser un número entre 0.0 y 1.0")
        return
    
    if not validate_float(iou_threshold, 0.0, 1.0):
        messagebox.showerror("Error de Validación", "IOU Threshold debe ser un número entre 0.0 y 1.0")
        return
    
    if not validate_float(box_thickness, 0.1, 10.0):
        messagebox.showerror("Error de Validación", "Box Thickness debe ser un número entre 0.1 y 10.0")
        return
    
    if not validate_float(font_size, 0.1, 5.0):
        messagebox.showerror("Error de Validación", "Font Size debe ser un número entre 0.1 y 5.0")
        return
    
    if not validate_int(webcam_path, 0, 10):
        messagebox.showerror("Error de Validación", "Webcam Path debe ser un entero entre 0 y 10")
        return

    # Update status
    status_label.config(text="Estado: Ejecutando script...", foreground="blue")
    root.update()

    # Build the command to call live_script.py with the configured arguments
    command = ['python', 'live_script.py',
               '--inference_size', inference_size,
               '--conf_threshold', conf_threshold,
               '--iou_threshold', iou_threshold,
               '--box_thickness', box_thickness,
               '--font_size', font_size,
               '--webcam_path', webcam_path]

    # Execute the command in a subprocess
    try:
        subprocess.run(command)
        status_label.config(text="Estado: Script completado exitosamente", foreground="green")
        print("Script executed with the options configured from the GUI.")
    except Exception as e:
        status_label.config(text=f"Estado: Error - {str(e)}", foreground="red")
        messagebox.showerror("Error", f"Error al ejecutar el script: {str(e)}")

# Create the main window of the GUI
root = tk.Tk()
root.title("YOLO Cell Count - Configuración de Parámetros")
root.geometry("500x600")
root.resizable(True, True)

# Add a style for better appearance
style = ttk.Style()
style.theme_use('clam')

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

# Main container with padding
main_frame = ttk.Frame(root, padding="15")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Title label
title_label = ttk.Label(main_frame, text="Configuración de Parámetros YOLO", 
                        font=('Arial', 14, 'bold'))
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

# Camera detection section
camera_frame = ttk.LabelFrame(main_frame, text="Cámaras Detectadas", padding="10")
camera_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))

if detected_cameras:
    camera_info = ttk.Label(camera_frame, text=f"Cámaras disponibles: {', '.join(map(str, detected_cameras))}")
else:
    camera_info = ttk.Label(camera_frame, text="No se detectaron cámaras", foreground="red")
camera_info.pack()

# Parameters section
params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Inferencia", padding="10")
params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))

# Create a helper function to add parameter rows
def add_parameter_row(parent, row, label_text, variable, tooltip_text):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, sticky=tk.W, pady=5, padx=(0, 10))
    
    entry = ttk.Entry(parent, textvariable=variable, width=15)
    entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
    
    # Tooltip label
    tooltip = ttk.Label(parent, text=tooltip_text, font=('Arial', 8), foreground='gray')
    tooltip.grid(row=row, column=2, sticky=tk.W, pady=5, padx=(10, 0))
    
    return entry

# Add all parameters with tooltips
inference_size_entry = add_parameter_row(params_frame, 0, "Inference Size:", 
                                         inference_size_var, "(320-1280)")
conf_entry = add_parameter_row(params_frame, 1, "Confidence Threshold:", 
                               conf_var, "(0.0-1.0)")
iou_entry = add_parameter_row(params_frame, 2, "IOU Threshold:", 
                              iou_var, "(0.0-1.0)")
box_thickness_entry = add_parameter_row(params_frame, 3, "Box Thickness:", 
                                        box_thickness_var, "(0.1-10.0)")
font_size_entry = add_parameter_row(params_frame, 4, "Font Size:", 
                                    font_size_var, "(0.1-5.0)")
webcam_path_entry = add_parameter_row(params_frame, 5, "Webcam ID:", 
                                      webcam_path_var, "(0-10)")

# Configure column weights for better resizing
params_frame.columnconfigure(1, weight=1)

# Buttons section
button_frame = ttk.Frame(main_frame)
button_frame.grid(row=3, column=0, columnspan=2, pady=20)

reset_button = ttk.Button(button_frame, text="Restablecer Valores Predeterminados", 
                          command=reset_default_values, width=30)
reset_button.grid(row=0, column=0, padx=5, pady=5)

run_button = ttk.Button(button_frame, text="Ejecutar Script", 
                        command=run_script, width=30)
run_button.grid(row=1, column=0, padx=5, pady=5)

# Status label
status_label = ttk.Label(main_frame, text="Estado: Listo", 
                        font=('Arial', 10), foreground="green")
status_label.grid(row=4, column=0, columnspan=2, pady=(10, 0))

# Instructions section
instructions_frame = ttk.LabelFrame(main_frame, text="Instrucciones", padding="10")
instructions_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 0))

instructions_text = """1. Configure los parámetros según sus necesidades
2. Haga clic en "Ejecutar Script" para iniciar la detección
3. En la ventana de detección:
   - Presione 'q' para salir
   - Presione 'i' para activar/desactivar inferencia
   - Presione 'm' para colores multicolor
   - Presione 'c'/'v' para ajustar contraste
   - Presione 'b'/'n' para ajustar brillo
   - Presione 'p' para tomar captura de pantalla"""

instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                              justify=tk.LEFT, font=('Arial', 9))
instructions_label.pack()

# Main function of the GUI
root.mainloop()
