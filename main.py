import time
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utiliz import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from vehicle_class import Vehicle








#veribles
vehicle_dict = {}

#insert video
# video_path = input("give the video path")
#load model
detact_vechicels_model =  (YOLO('yolov10s.pt')).to(device)
track_vechicels_model = DeepSort(max_age=30, embedder_gpu=True)

#load video
video_path = '/workspace/data/video2.mp4'
cap = cv2.VideoCapture(video_path)

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1 / fps



# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
   
#read frame
while True:
    start_time = time.time() 
    ret, frame = cap.read()
    
    # If the frame was not grabbed (end of video), break the loop
    if not ret:
        print("End of video")
    start_time1 = time.time()
    # Step 1: vechicels Detection with YOLOv10
    outputs = detact_vechicels_model(frame)  # Perform inference on the frame
    
    # Step 2: Extract bounding boxes, confidences, and class ids
    update_input = input_for_update_tracks(outputs)
    
    # Step 3: Update Deep SORT tracker
    vechicels  = track_vechicels_model.update_tracks(update_input, frame=frame) 
    
    
    elapsed_time1 = time.time() - start_time1
    # Step 4: Extract vechicels and set them
    start_time2 = time.time() 
    for vechicel in vechicels:
        vechicel_id = int(vechicel.track_id)

        # Check if the vehicle is deleted
        if vechicel.is_deleted():
            vehicle_dict.pop(vechicel_id)
            continue

        if vechicel.is_confirmed():

            #if the vehicle is new create a new instance and add it to the dict
            if vechicel_id not in vehicle_dict :
                cur_instatnce = Vehicle(vechicel,frame)
                vehicle_dict[vechicel_id] = cur_instatnce

            #if the vehicle is not new update the instance
            if vechicel_id in vehicle_dict:
                cur_instatnce = vehicle_dict[vechicel_id]
                cur_instatnce.update(vechicel, frame)
            

            frame =   cur_instatnce.draw_vehicle(frame)
        
    elapsed_time2 = time.time() - start_time2
        
    #צריך לכתוב קוד שנותן את 4 הבוקסים הטובים ביותר
    # Step 5: sort the dict by the confidence of the plate number and pick the top 4
    # for 
    # sort_list = sort_vehicle_dict(vehicle_dict)
    # for best_veh in sort_list[:4]:
    #     frame =  best_veh.draw_vehicle(frame)
        

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    wait_time = max(1, int((frame_interval - elapsed_time) * 1000))  # Wait in milliseconds
    # Release the video capture object and close all windows
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
     
    
cap.release()
cv2.destroyAllWindows()

