from utiliz import *
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detact_licence_plates = (YOLO('license_plate_detector.pt',verbose=False)).to(device)



class Vehicle:
    def __init__(self, vehicle,frame):
        """
        Initialize a Vehicle object.

        :param vehicle_id: Unique identifier for the vehicle.
        :param bounding_box: Bounding box coordinates of the vehicle.
        :param plate_number: License plate number of the vehicle (default is None).
        """
        self.vehicle_id = vehicle.track_id
        self.vehicle_bounding_box = list(map(int,vehicle.to_tlbr()))
        self.plate_bbox = None
        self.plate_dict = {}
        self.plate_number = "unknown"
        self.plate_conf = 0
        
    def update_bounding_box(self, vehicle):
        """
        Update the bounding box of the vehicle.

        :param new_bounding_box: New bounding box coordinates.
        """
        self.vehicle_bounding_box= list(map(int,vehicle.to_tlbr()))
        

    def detact_plate_bbox(self,frame):
        bb_box = crop_bb(self.vehicle_bounding_box,frame)
        detact_plate = detact_licence_plates(bb_box,verbose=False)
        plate_box = detact_plate[0].boxes.xyxy.tolist()
        if  len(plate_box) == 0 :
            self.plate_bbox = None
            return
        plate_box = [int(value) for value in plate_box[0]]
        self.plate_bbox = plate_box

    def update_plate_dict(self,plate_number,confidence):
        """
        Update the plate dictionary with the new plate number.

        :param plate_number: License plate number.
        """
        
        plate_number = "".join(plate_number)

        # Update the plate dictionary with the new plate number
        if plate_number in list(self.plate_dict.keys()):
            self.plate_dict[plate_number] += confidence
        else:
            self.plate_dict[plate_number] = confidence

        # Sort the plate dictionary by confidence values , just 3 plate numbers
        self.plate_dict = dict(sorted(self.plate_dict.items(), key=lambda item: item[1], reverse=True)[:3])

        # Return the most confident plate number
        conf_number = (next(iter(self.plate_dict))) 
        confidence = float(self.plate_dict[conf_number])
        if confidence < 0.45:
            conf_number = "unknown"
            confidence = 0
        return conf_number , confidence
    

        


    def update_plate_number(self,frame):
        """
        Set the license plate number of the vehicle.

        :param plate_number: License plate number.
        """
        # Read the license plate number
        if self.plate_bbox is None:
            self.plate_number = "unknown"
            return
        Vehicle_img = crop_bb(self.vehicle_bounding_box,frame)
        plate_number,confidence = extract_plate_number(self.plate_bbox,Vehicle_img)
        if plate_number is None:
            return
        self.plate_number,self.plate_conf = self.update_plate_dict(plate_number,confidence)
     
    def update(self,vehicle,frame):
        """
        Update the vehicle object.

        :param new_bounding_box: New bounding box coordinates.
        :param plate_number: License plate number.
        """
        self.update_bounding_box(vehicle)
        if self.plate_conf>1:
            return
        self.detact_plate_bbox(frame)
        self.update_plate_number(frame)
        

               
   
    def show(self,frame):#show me a frame of the vehicle with the predicted plate number
        vehicle_img = crop_bb(self.vehicle_bounding_box,frame)
        plate_img = crop_bb(self.plate_bbox,vehicle_img)
            # Convert the plate image to grayscale

        #show the image 
        if vehicle_img is not None:
            cv2.imshow('vehicle',vehicle_img)
        if plate_img is not None:
            cv2.imshow('plate',plate_img)
        if cv2.waitKey(30) & 0xFF == ord('r'):
            cv2.destroyWindow('vehicle+plate')
        

    def draw_vehicle(self, frame):
        # Unpacking the vehicle bounding box coordinates
        x1, y1, x2, y2 = self.vehicle_bounding_box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Blue color for rectangle and text background
        rect_color = (255, 87, 34)  # Amber color for rectangle
        text_bg_color = (30, 144, 255)  # Dodger blue for text background

        # Draw a thicker rectangle for the vehicle
        thickness = 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, thickness)

        # Plate number text
        text = self.plate_number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Text background rectangle (with slight offset for better appearance)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), text_bg_color, -1)

        # Put text with a slight shadow
        shadow_offset = 2
        cv2.putText(frame, text, (x1 + shadow_offset, y1 - 5 + shadow_offset), font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

        return frame




        
