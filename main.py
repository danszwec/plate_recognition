import time
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utiliz import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import easyocr
from vechile_class import Vehicle

#veribles
vechicel_dict = {}

#insert video
# video_path = input("give the video path")
#load model
detact_vechicels_model =  (YOLO('yolov10s.pt')).to(device)
track_vechicels_model = DeepSort(max_age=30, embedder_gpu=True)

#load video
cap = cv2.VideoCapture('/home/oury/Documents/dan/projects/data/video/[appsgolem.com][00-27-00][00-28-00]_Driving_from_Ben_Gurion_Airpor.mp4')

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

        if vechicel.is_deleted():
            vechicel_dict.pop(vechicel_id)

        if vechicel_id not in vechicel_dict and vechicel.is_confirmed():
            cur_instatnce = Vehicle(vechicel,frame)
            vechicel_dict[vechicel_id] = cur_instatnce

        if vechicel_id in vechicel_dict:
            vechicel_dict[vechicel_id].update(vechicel,frame)
        
        

    #read licence plate numberplate_img
        if vechicel_id == 1:
            cv2.imshow("plate",plate_img)
        plate_number,confidence = extract_plate_number(plate_img,reader)
        conf_lst = update_confidence_list(vechicel_id,confidence,plate_number,vechicel_dict)
        if (vechicel_dict[vechicel_id])[1] < 0.6:
            plate_number = "Unkown"
        else :
            plate_number = (vechicel_dict[vechicel_id])[0]
            # cv2.imshow(str(vechicel_id),plate_img)
    
    #draw on frame
        frame = draw_vechicel(frame, vechicel ,plate_number)

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

