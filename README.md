# License Plate Recognition System
This project implements a Cars Tracker that effectively recognizes the license plates of tracked vehicles in real time. Designed for a camera mounted in a car, the system records and detects vehicles in the surroundings, recognizing their license plates to provide evidence in the event of an accident.
 
Utilizing [**YOLOv8**](https://github.com/autogyro/yolo-V8) for accurate vehicle detection,  [**SORT**](https://pypi.org/project/sort-tracker/) for robust tracking across frames, and [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) for reading and recognizing license plate text, this project aims to enhance vehicle tracking capabilities and streamline the process of license plate recognition in various environments.


### Ongoing Enhancements:
1. Increasing FPS for Real-Time Performance:
Currently, we are working on optimizing the system's performance by training and customizing YOLO to focus specifically on vehicle detection. This aims to reduce computational overhead and achieve higher frame rates (FPS) without sacrificing detection accuracy.

2. Improving License Plate Recognition Accuracy:
The project has already transitioned from EasyOCR to PaddleOCR for license plate text recognition. This change significantly improved recognition accuracy, and we continue to fine-tune and enhance this component for even better results in various real-world scenarios.

[![Watch the video][![Watch the video](https://github.com/danszwec/plate_recognition/edit/master/first_10_seconds.mp4)](https://github.com/danszwec/plate_recognition/edit/master/first_10_seconds.mp4)


*Note: The entire video was processed for detection and saving frame-by-frame, followed by applying slow motion at 50% speed. The actual code is designed to work at an average of 25 FPS.*


## Installation Instructions with Docker 


Clone the repository
 ```bash
 git clone https://github.com/yourusername/cars-tracker.git
```

Navigate to the project directory
```bash
 cd plate_recognizer
```
Build the Docker image
```bash
 docker build -t plate_recognizer
```
Run the Docker container
```bash
 docker run --rm -it -v /path/to/your/data:/data cars-tracker
 ```
     
**Note**: This is not the final version of the project; it is a work in progress, and future updates will further enhance its capabilities and performance.


