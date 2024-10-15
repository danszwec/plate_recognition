import os
from ultralytics import YOLO

# Set paths
data_yaml = '/home/oury/Documents/dan/projects/plate_recognition/data/data.yaml'
pretrained_weights = "yolov10s.pt"  # Use the pretrained weights for YOLOv11
epochs = 50  # Adjust the number of epochs as needed

# Create the YOLO model
model = YOLO(pretrained_weights)  # Load the YOLOv11 model (adjust to the right path if needed)

# Train the model
model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16)

# Optionally save the model after training
model.save("israel_license_plate_detector.pt")
