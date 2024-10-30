import cv2
import time
import torch
from PIL import Image
import numpy as np


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
        



def crop_bb(vechicel,frame):
    
    if isinstance(vechicel,torch.Tensor):
        x1, y1, x2, y2 = vechicel[0]
    else:
        x1, y1, x2, y2 = vechicel.to_tlbr()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    bb_img = frame[int(y1):int(y2), int(x1):int(x2)]
    
    return bb_img

def convert_to_grey(img,sup_thresh,value_apply):
    # Convert the plate image to grayscale
    gray_plate_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to convert pixels above  the thresholed to white
    _, thresholded_img = cv2.threshold(gray_plate_img, sup_thresh,value_apply, cv2.THRESH_BINARY)

    return thresholded_img

def extract_plate_number(plate_img,reader):
  
    # Read the license plate number
    result = reader.readtext(plate_img)
    if len(result) == 0:
        return None,0
    plate_number = [char for char in result[0][1] if char.isdigit()]
    if len(plate_number) != 7: #only 7 digits
        return None,0
    confidence = float(result[0][2])    
    return plate_number,confidence

def draw_bb(frame,outputs):
    for output in outputs:
        x1, y1, x2, y2 = map(int, (output.boxes.xyxy).squeeze())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
    

def update_confidence_list(vechicel_id,confidence,plate_number,conf_dict): 
    
# Update the confidence list with the new confidence values
    if vechicel_id  in conf_dict.keys():
        if (conf_dict[vechicel_id])[1] > 2:
            return conf_dict
        else:
            if isinstance(plate_number,str) == False:
                return conf_dict
           
            if (conf_dict[vechicel_id])[0] == plate_number:
                (conf_dict[vechicel_id])[1] += confidence

    else:
        conf_dict[vechicel_id] = [plate_number,confidence]
    
    return conf_dict

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


def update_plate_dict(plate_number,confidence,plate_dict):
    #sperete the plate number to digit and check which digit is the most confident
        for i in range(len(plate_number)):
            if plate_dict.get(i) is None:
                plate_dict[i] = []
            for j in range(len(plate_dict[i][0])):
                #if the the digit already exist in the place accumlate the confidence
                if plate_dict[i][0][j][0] == plate_number[i]:
                    plate_dict[i][0][j][1] += confidence
                    #sort the list by confidence
                    plate_dict[i][0]=sorted(plate_dict[i][0], key=lambda x: x[1], reverse=True)

                    match_found = True 
            
            if match_found:
                match_found = False
                continue
            #if the digit is not in the plate add it to the dict
            if len(plate_dict[i][0]) < 2:
                plate_dict[i][0].append([plate_number[i],confidence])

                #sort the list by confidence
                plate_dict[i][0]=sorted(plate_dict[i][0], key=lambda x: x[1], reverse=True)

            if len(plate_dict[i][0]) == 2:
                #find the index of the min confidence
                index_of_min = min(range(2), key=lambda j: plate_dict[i][0][j][1])

                #if the confidence of the new plate is higher than the min confidence replace the min confidence with the new plate    
                if plate_dict[i][0][index_of_min][1] < confidence:
                    plate_dict[i][0].replace(plate_dict[i][0][index_of_min],[plate_number[i],confidence])
                    
                    #sort the list by confidence
                    plate_dict[i]=sorted(plate_dict[i], key=lambda x: x[1], reverse=True)
            
        return plate_dict

def compose_plate_number(plate_dict):
    #compose the plate number
    plate_number = ''
    for i in range(len(plate_dict)):
        if plate_dict.get(i) is None:
            return None
        plate_number += plate_dict[i][0][0][0]
    return plate_number
