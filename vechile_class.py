from utiliz import *
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detact_licence_plates = (YOLO('license_plate_detector.pt')).to(device)
reader = easyocr.Reader(['en'])
detact_licence_plates = (YOLO('license_plate_detector.pt')).to(device)



class Vehicle:
    def __init__(self, vehicle,frame):
        """
        Initialize a Vehicle object.

        :param vehicle_id: Unique identifier for the vehicle.
        :param bounding_box: Bounding box coordinates of the vehicle.
        :param plate_number: License plate number of the vehicle (default is None).
        """
        self.vehicle_id = vehicle.track_id
        self.bounding_box = crop_bb(vehicle,frame)
        self.plate_dict = {}
        self.plate_number = plate_number
        

    def update_bounding_box(self, vehicle,frame):
        """
        Update the bounding box of the vehicle.

        :param new_bounding_box: New bounding box coordinates.
        """
        new_bounding_box = crop_bb(vehicle,frame)
        self.bounding_box = new_bounding_box

    def update_plate_dict(self, bb_box):
        """
        Set the license plate number of the vehicle.

        :param plate_number: License plate number.
        """
        detact_plate = detact_licence_plates(bb_box)
        box = detact_plate[0].boxes.xyxy
        if len(box) == 0:
            return None
        plate_img = crop_bb(box,bb_box)  #crop licence plates

        # Convert the plate image to grayscale for better OCR results
        plate_img = convert_to_grey(plate_img,125,255)

        plate_number,confidence = extract_plate_number(plate_img,reader)
        

        לתקן את זההההה
        #sperete the plate number to digit and check which digit is the most confident
        # for i in range(len(plate_number)):
        #     if len(self.plate_dict[i]) < 2:
        #         self.plate_dict[i].append([plate_number[i],confidence])
        #     if len(self.plate_dict[i]) == 2:
        #         min_conf = min(self.plate_dict[i][0][1],self.plate_dict[i][1][1])
        #         if min_conf < confidence:
        #             self.plate_dict[i].replace(min_conf,[plate_number[i],confidence])

                    
              
                    
    def get_info(self):
        """
        Get the information of the vehicle.

        :return: Dictionary containing vehicle information.
        """
        return {
            'vehicle_id': self.vehicle_id,
            'bounding_box': self.bounding_box,
            'plate_number': self.plate_number
        }

    def __str__(self):
        """def update_plate_number(self, plate_number):
        String representation of the Vehicle object.

        :return: String containing vehicle information.
        """
        return f"Vehicle ID: {self.vehicle_id}, Bounding Box: {self.bounding_box}, Plate Number: {self.plate_number}"
    
    def update(self, vehicle,frame):
        """
        Update the vehicle object.

        :param new_bounding_box: New bounding box coordinates.
        :param plate_number: License plate number.
        """
        self.update_bounding_box(vehicle,frame)
        self.update_plate_number(plate_number)