import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utiliz import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# יום שלישי #
#להוריד סרטונים
#לראות שהקוד עובד
#לבנות דוקר
#לעלות לגיט

#insert video
# video_path = input("give the video path")
vechicels_yolo_classes = [2,3,5,7]

#load model
detact_vechicels_model =  (YOLO('yolov10s.pt')).to(device)
track_vechicels_model = DeepSort(max_age=30, embedder_gpu=True)
detact_licence_plates = (YOLO('license_plate_detector.pt')).to(device)


#load video
cap = cv2.VideoCapture('/home/oury/Documents/dan/projects/plate_recognition/data/video/[appsgolem.com][00-27-00][00-28-00]_Driving_from_Ben_Gurion_Airpor.mp4')

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# captue rhe inital time
start_time = time.perf_counter()
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
   
#read frame
while True: 
    ret, frame = cap.read()
    # If the frame was not grabbed (end of video), break the loop
    if not ret:
        print("End of video")
    
    # Step 1: vechicels Detection with YOLOv10
    outputs = detact_vechicels_model(frame)  # Perform inference on the frame

    # Step 2: Extract bounding boxes, confidences, and class ids
    update_input = input_for_update_tracks(outputs)

    # Step 3: Update Deep SORT tracker
    vechicels  = track_vechicels_model.update_tracks(update_input, frame=frame) # Update Deep SORT tracker
    
    # Step 4: Extract vechicels that are confirmed
    for vechicel in vechicels:
        if not vechicel.is_confirmed():
            continue    
        vechicel_img = crop_bb(vechicel,frame)
        if 0 in vechicel_img.shape:
            continue
    # Step 5: Detect license plates
        detact_plate = detact_licence_plates(vechicel_img)
        box = detact_plate[0].boxes.xyxy
        if len(box) == 0:
            continue
        plate_img = crop_bb(box,vechicel_img)  #crop licence plates 
        cv2.imwrite('output_image2.jpg', plate_img)

    #read licence plate numberplate_img
        plate_number,confidence = extract_plate_number(plate_img)
        if confidence < 0.6:
            plate_number = "unknown"
    
    #draw on frame
        frame = draw_vechicel(frame, vechicel ,plate_number)

    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    # Calculate the elapsed time
    elapsed_time = time.perf_counter() - start_time
    adjusted_delay = max(1, int(1000 / fps) - int(elapsed_time * 1000)) 
# Release the video capture object and close all windows
    if cv2.waitKey(int(elapsed_time)) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()

