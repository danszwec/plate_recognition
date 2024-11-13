import time
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utiliz import *
from sort import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from vehicle_class import Vehicle
from time_manage import TimeManager as tm
import numpy as np


# צריך:
# 1 לראות שלא עושים כפילויות
# 2.  לאמן את הקורא לוחיות אולי RNN
# 3.לקצר את העדכון של הרכבים . רק רכבים שזיהנו פלייט



track_vehicles_model = Sort(max_age=3, min_hits=1, iou_threshold=0.5)
#veribles
vehicle_dict = {}

#time manager
time_manager = tm()

#load model
detect_vehicles_model =  (YOLO('yolov8n.pt',verbose=False)).to(device)


#load video
video_path = '/workspace/data/video2.mp4'
# video_path = 'C:/Users/danha/ai_projects/plate_recognition/east_jerusalem.mp4'
cap = cv2.VideoCapture(video_path)

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1 / fps
num_frame = 0
elapsed_time_acc = 0
elapsed_time2_acc = 0
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
   
#read frame
while True:
    time_manager.clear()
    ret, frame = cap.read()
    time.time()
    num_frame += 1
    # If the frame was not grabbed (end of video), break the loop
    if not ret:
        print("End of video")
    start_time = time.time() 
    
    # Step 1: vehicles Detection with YOLOv8n
    time_manager.start('yolov8n inference')
    outputs = detect_vehicles_model(frame,verbose=False)  # Perform inference on the frame
    time_manager.stop('yolov8n inference')


    
    # Step 2: Extract bounding boxes, confidences, and class ids just for vehicles with plates
    time_manager.start('yolo outputs to trecker input')
    update_input = input_for_update_sort(outputs,frame)
    time_manager.stop('yolo outputs to trecker input')
    
    # Step 3: Update Deep SORT tracker
    time_manager.start('update tracks') 
    # vehicles  = track_vehicles_model.update(update_input) 
    vehicles  = track_vehicles_model.update(update_input) 
    time_manager.stop('update tracks')

    # Step 4: Extract vehicles and set them
    time_manager.start('update all the vehicles')
    for vehicle in vehicles:
        time_manager.start('update one vehicle')
        vehicle_id = str(vehicle[4])
        #if the vehicle is new create a new instance and add it to the dict
        if vehicle_id not in vehicle_dict :
            cur_instatnce = Vehicle(vehicle,frame)
            vehicle_dict[vehicle_id] = cur_instatnce

        #if the vehicle is not new update the instance
        if vehicle_id in vehicle_dict:
            cur_instatnce = vehicle_dict[vehicle_id]
            cur_instatnce.update(vehicle, frame)
        
        if cur_instatnce.plate_conf != 0:
            frame = cur_instatnce.draw_vehicle(frame)
        time_manager.stop('update one vehicle')
        # if cur_instatnce.vehicle_id == str(7):
        #     cur_instatnce.show(frame)

    #sort the vehicles by confidence
    vehicle_dict = sort_vehicle_dict(vehicle_dict)

    time_manager.stop('update all the vehicles')

    # Display the resulting frame
    cv2.imshow('frame', frame)
    

    # Calculate the elapsed time
    elapsed_time_acc += (time.time() - start_time)
    elapsed_time = (elapsed_time_acc) / num_frame

    
    time_manager.summary(num_frame,show=True)

    
    wait_time = max(1, int((frame_interval - elapsed_time) * 1000))  # Wait in milliseconds
    # Release the video capture object and close all windows
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
     

cap.release()
cv2.destroyAllWindows()

