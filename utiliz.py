import cv2
import easyocr
import torch


def input_for_update_tracks(outputs):
    # Extract yolov10s output to input for update tracks
    update = []
    for output in outputs[0]:
        boxes = output.boxes.xyxy  
        confidences = output.boxes.conf 
        label = output.boxes.cls
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])  
            conf = float(confidences[i])
            cur_tuple = ([x1, y1, x2 - x1, y2 - y1], conf,int(label[i]))
            update.append(cur_tuple)
    return update
        
    


def extract_classes_bbox(outputs,classes):
#extract bboxs that menthon on classes list 
    for output in outputs:
        if output['class'] not in classes:
            outputs.remove(output)
    return outputs


def crop_bb(vechicel,frame):
    
    if isinstance(vechicel,torch.Tensor):
        x1, y1, x2, y2 = vechicel[0]
    else:
        x1, y1, x2, y2 = vechicel.to_tlbr()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    bb_img = frame[int(y1):int(y2), int(x1):int(x2)]
    return bb_img

def draw_vechicel(frame, vechicel ,plate_number):
    x1, y1, x2, y2 = vechicel.to_tlbr()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, plate_number, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def extract_plate_number(plate_img):

    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate_img)
    if len(result) == 0:
        return None,0
    plate_number = result[0][1]
    confidence = result[0][2]
    return plate_number,confidence

def draw_bb(frame,outputs):
    for output in outputs:
        x1, y1, x2, y2 = map(int, (output.boxes.xyxy).squeeze())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
    




