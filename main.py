import time
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utiliz import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import easyocr
from vechile_class import Vehicle

להציג רק את הטובים ביותר
אם הפלייט לא מעל 0.8 לא להציג
צבעים להצגה 







#veribles
vechicel_dict = {}

#insert video
# video_path = input("give the video path")
#load model
detact_vechicels_model =  (YOLO('yolov10s.pt')).to(device)
track_vechicels_model = DeepSort(max_age=30, embedder_gpu=True)

#load video
video_path = 'east_jerusalem.mp4'
cap = cv2.VideoCapture(video_path)

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1 / fps

# captue rhe inital time

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
    
    # Step 1: vechicels Detection with YOLOv10
    outputs = detact_vechicels_model(frame)  # Perform inference on the frame
    
    # Step 2: Extract bounding boxes, confidences, and class ids
    update_input = input_for_update_tracks(outputs)
    
    # Step 3: Update Deep SORT tracker
    vechicels  = track_vechicels_model.update_tracks(update_input, frame=frame) \
    
    # Step 4: Extract vechicels and set them
    for vechicel in vechicels:
        vechicel_id = int(vechicel.track_id)

        # Check if the vehicle is deleted
        if vechicel.is_deleted():
            vechicel_dict.pop(vechicel_id)

        
        if vechicel.is_confirmed():

            #if the vehicle is new create a new instance and add it to the dict
            if vechicel_id not in vechicel_dict :
                cur_instatnce = Vehicle(vechicel,frame)
                vechicel_dict[vechicel_id] = cur_instatnce

            #if the vehicle is not new update the instance
            if vechicel_id in vechicel_dict:
                cur_instatnce = vechicel_dict[vechicel_id]
                cur_instatnce.update(vechicel, frame)

        

    #read licence plate numberplate_img
            if vechicel_id == 1:
                cur_instatnce.show()
    #draw on frame
            frame =  cur_instatnce.draw_vechicel(frame)
        

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

