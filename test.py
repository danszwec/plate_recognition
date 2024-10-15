import cv2
def show_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 40
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    while True: 
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if not ret:
            print("End of video")
            break
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
q    show_video('/home/oury/Documents/dan/projects/plate_recognition/data/video/Driving_from_Ben_Gurion_Airpor.avi')