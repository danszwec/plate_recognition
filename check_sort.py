import time
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sort import *
from utiliz import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from vehicle_class import Vehicle
from time_manage import TimeManager as tm


# צריך:
# 1 לעבור לסורט ולא לדיפ סורט
# 2. לאמן את הקורא לוחיות
# 3.לקצר את העדכון של הרכבים . רק רכבים שזיהנו פלייט
# 4. אם אנחנו בטוחים בפלייט לא להתחיל לחפש



track_vehicles_model = Sort(max_age=3, min_hits=1, iou_threshold=0.5)
#veribles
vehicle_dict = {}

#time manager
time_manager = tm()

#load model
detact_vehicles_model =  (YOLO('yolov8n.pt',verbose=False)).to(device)
# track_vehicles_model = DeepSort(max_age=2,nms_max_overlap=0.5, embedder_gpu=True)

#load video
video_path = '/workspace/data/video2.mp4'
# video_path = 'C:/Users/danha/ai_projects/plate_recognition/east_jerusalem.mp4'
cap = cv2.VideoCapture(video_path)

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1 / fps
num_frame = 0

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
    outputs = detact_vehicles_model(frame,verbose=False)  # Perform inference on the frame
    time_manager.stop('yolov8n inference')


    
    # Step 2: Extract bounding boxes, confidences, and class ids just for vehicles with plates
    time_manager.start('yolo outputs to trecker input')
    # update_input = input_for_update_deepsort(outputs,frame)
    update_input = input_for_update_sort(outputs,frame)
    time_manager.stop('yolo outputs to trecker input')
    
    # Step 3: Update Deep SORT tracker
    time_manager.start('update tracks') 
    vehicles  = track_vehicles_model.update(update_input) 
    time_manager.stop('update tracks')
    time_manager.summary(num_frame,show=True)
    # Step 4: Extract vehicles and set them
    time_manager.start('update all the vehicles')
    for vehicle in vehicles:
        vehicle_id = vehicle[4]
        x1, y1, x2, y2 = map(int, vehicle[:4])
        #draw the vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #add the id on frame
        cv2.putText(frame, str(vehicle_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    #show the frame
    cv2.imshow('frame', frame)

    #finish the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break