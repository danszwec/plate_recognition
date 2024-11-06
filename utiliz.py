import cv2
import time
import torch
from PIL import Image
import easyocr
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reader = easyocr.Reader(['en'], gpu=True)


def input_for_update_tracks(outputs):
    # Extract yolov10s output to input for update tracks with the vehicle class
    vehicles_classes = [2,3,5,7]
    update = []
    for output in outputs[0]:
        boxes = output.boxes.xyxy  
        confidences = output.boxes.conf 
        label = output.boxes.cls
        for i in range(len(boxes)):
            if int(label[i]) in vehicles_classes:
                x1, y1, x2, y2 = map(int, boxes[i])  
                conf = float(confidences[i])
                cur_tuple = ([x1, y1, x2 - x1, y2 - y1], conf,int(label[i]))
                update.append(cur_tuple)
    return update
        



def crop_bb(bbox,frame):
    x1, y1, x2, y2 = bbox
    bb_img = frame[int(y1):int(y2), int(x1):int(x2)]
    if bb_img.size ==  0:
        return None
    
    return bb_img


def extract_plate_number(bbox,frame):

    plate_img = crop_bb(bbox,frame)
    if plate_img is None:
        return  None,0

    # Convert the plate image to grayscale
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to convert pixels above  the thresholed to white
    _, thresholded_img = cv2.threshold(gray_plate_img, 50, 255, cv2.THRESH_BINARY)
    final_img = cv2.resize(thresholded_img, (100, 100))
    cv2.imshow('plate',final_img)
  
    # Read the license plate number
    result = reader.readtext(thresholded_img)
    if len(result) == 0:
        return None,0
    plate_number = [char for char in result[0][1] if char.isdigit()]
    if len(plate_number) != (7 or 8) : #only 7 digits
        return None,0
    confidence = float(result[0][2])    
    return plate_number,confidence

def draw_bb(frame,outputs):
    for output in outputs:
        x1, y1, x2, y2 = map(int, (output.boxes.xyxy).squeeze())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
    


def resize_plate(plate_img):
    # Calculate new size while maintaining aspect ratio
    original_size = plate_img.shape  # (width, height)
    aspect_ratio = original_size[0] / original_size[1]

    # New height or width
    new_width = 128
    new_height = int(new_width / aspect_ratio)

    # Resize the image
    plate_img = Image.fromarray(plate_img)
    resized_image = plate_img.resize((new_height,new_width))
    resized_image = np.array(resized_image)

    return resized_image

def sort_vehicle_dict(input_dict):
    # Sort the dictionary keys by the .self_conf attribute of the items
    sorted_vechicles = sorted(input_dict.values(), key=lambda x: x.plate_conf, reverse=True)
    return sorted_vechicles



