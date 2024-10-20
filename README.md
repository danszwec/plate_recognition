# License Plate Recognition System
This project implements a Cars Tracker that effectively recognizes the license plates of tracked vehicles in real time. Designed for a camera mounted in a car, the system records and detects vehicles in the surroundings, recognizing their license plates to provide evidence in the event of an accident.
 
Utilizing [**YOLOv10**](https://github.com/THU-MIG/yolov10) for accurate vehicle detection,  [**DeepSORT**](https://pypi.org/project/deep-sort-realtime/) for robust tracking across frames, and EasyOCR for reading and recognizing license plate text, this project aims to enhance vehicle tracking capabilities and streamline the process of license plate recognition in various environments.






## Installation Instructions with Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/cars-tracker.git

```bash
#Navigate to the project directory
cd plate_recognizer

```bash
#Build the Docker image
docker build -t plate_recognizer

```bash
#Run the Docker container
docker run --rm -it -v /path/to/your/data:/data cars-tracker

    



