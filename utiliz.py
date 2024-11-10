import cv2
import time
import torch
from PIL import Image
import easyocr
import numpy as np
import pytesseract
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reader = easyocr.Reader(['en'], gpu=True)



def input_for_update_tracks(outputs):
    """
    Optimized function to process YOLOv8 outputs for vehicle tracking.
    
    Parameters:
    outputs: YOLOv8 detection outputs
    
    Returns:
    List[Tuple]: List of tuples containing (bbox, confidence, class_id)
    """
    # Use set for faster lookup
    VEHICLES_CLASSES = {2, 3, 5, 7}
    
    # Get references to avoid repeated attribute access
    boxes = outputs[0].boxes
    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls
    
    # Pre-calculate array length
    length = len(xyxy)
    
    # Pre-allocate list with estimated size
    update = []
    update.reserve(length) if hasattr(list, 'reserve') else None  # Optional optimization for CPython implementations that support it
    
    # Vectorized mask for vehicle classes
    class_mask = np.isin(cls.cpu().numpy(), list(VEHICLES_CLASSES))
    
    # Get indices where class_mask is True
    valid_indices = np.where(class_mask)[0]
    
    # Process only valid indices
    for idx in valid_indices:
        x1, y1, x2, y2 = map(int, xyxy[idx])
        # Calculate width and height directly
        w = x2 - x1
        h = y2 - y1
        update.append(([x1, y1, w, h], float(conf[idx]), int(cls[idx])))
    
    return update



def crop_bb(bbox, frame):
    """
    Optimized function to crop a bounding box from a frame.
    
    Parameters:
    bbox: Tuple of (x1, y1, x2, y2) coordinates
    frame: Input image frame
    
    Returns:
    Optional[np.ndarray]: Cropped image or None if invalid
    """
    # Early return for None bbox
    if bbox is None:
        return None
    
    try:
        # Get frame dimensions once
        frame_height, frame_width = frame.shape[:2]
        
        # Convert coordinates to integers and bound them to frame dimensions
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame_width, int(bbox[2]))
        y2 = min(frame_height, int(bbox[3]))
        
        # Check if coordinates are valid
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Perform cropping with pre-validated coordinates
        return frame[y1:y2, x1:x2]
        
    except (IndexError, ValueError, AttributeError):
        return None


def extract_plate_number(bbox,frame):

    plate_img = crop_bb(bbox,frame)
    if plate_img is None:
        return  None,0

    # Convert the plate image to grayscale
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to convert pixels above  the thresholed to white
    _, thresholded_img = cv2.threshold(gray_plate_img, 50, 255, cv2.THRESH_BINARY)
    final_img = cv2.resize(thresholded_img, (100, 100))
  
    # Read the license plate number
    result = reader.readtext(thresholded_img)
    
    if len(result) == 0:
        return None,0
    plate_number = [char for char in result[0][1] if char.isdigit()]
    if len(plate_number) not in [7, 8]: # only 7 or 8 digits
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

def plate_exist(vehicle,frame):
    box = list(map(int,vehicle.to_tlbr()))
    vehicle
