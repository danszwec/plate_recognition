# License Plate Recognition System
This project implements a Cars Tracker that effectively recognizes the license plates of tracked vehicles in real time. Designed for a camera mounted in a car, the system records and detects vehicles in the surroundings, recognizing their license plates to provide evidence in the event of an accident.
 
Utilizing [**YOLOv8**](https://github.com/autogyro/yolo-V8) for accurate vehicle detection,  [**SORT**](https://pypi.org/project/sort-tracker/) for robust tracking across frames, and [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) for reading and recognizing license plate text, this project aims to enhance vehicle tracking capabilities and streamline the process of license plate recognition in various environments.








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


