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
        self.vehicle_bounding_box = vehicle.to_tlbr()
        self.plate_bbox = None
        self.plate_dict = {}
        self.plate_number = "unknown"
        
    def update_bounding_box(self, vehicle):
        """
        Update the bounding box of the vehicle.

        :param new_bounding_box: New bounding box coordinates.
        """
        self.vehicle_bounding_box= vehicle.to_tlbr()
        

    def update_plate_number(self,frame):
        """
        Set the license plate number of the vehicle.

        :param plate_number: License plate number.
        """
        bb_box = crop_bb(self.vehicle_bounding_box,frame)
        detact_plate = detact_licence_plates(bb_box)
        self.plate_bouding_box = detact_plate[0].boxes.xyxy.to_tlbr()
        if len(self.plate_bouding_box) == 0:
            self.plate_bouding_box = None
            return
        

        # Read the license plate number
        plate_number,confidence = extract_plate_number(self.bounding_box,reader)

        # Update the plate dict with the new plate number
        self.plate_dict = update_plate_dict(plate_number,confidence,self.plate_dict)

        # compose the plate number
        self.plate_number = compose_plate_number(self.plate_dict)

    def update(self, vehicle,frame):
        """
        Update the vehicle object.

        :param new_bounding_box: New bounding box coordinates.
        :param plate_number: License plate number.
        """
        self.update_bounding_box(vehicle,frame)
        self.bounding_box = vehicle.to_tlbr()
        self.update_plate_number(frame,self.bounding_box)
        
                
                    
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
        return f"Vehicle ID: {self.vehicle_id}, Bounding Box: {self.vehicle_bounding_box}, Plate Number: {self.plate_number}"
    
   
    def show(self,frame):#show me a frame of the vehicle with the predicted plate number
        vehicle_img = crop_bb(self.vehicle_bounding_box,frame)
        plate_img = crop_bb(self.plate_bouding_box,frame)
        hight, width, _ = self.vehicle_img.shape
        black_img = np.zeros((hight+50, width, 3), np.uint8)
        
        #put the boundings box on the black image
        black_img[0:hight, 0:width] = vehicle_img
        #put the plate img on the black image
        black_img[hight:hight+20, 0:width] = plate_img

        #put the plate number on the black image
        cv2.putText(black_img, self.plate_number, (0, hight+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #show the image 
        cv2.imshow('vehicle+plate', black_img)
        if cv2.waitKey(30) & 0xFF == ord('r'):
            cv2.destroyWindow('vehicle+plate')
        


    def draw_vechicel(self,frame):
        x1, y1, x2, y2 = self.bounding_box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, self.plate_number, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    


        
