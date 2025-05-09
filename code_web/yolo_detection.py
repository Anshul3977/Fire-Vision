import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

global_model = None

def load_model():
    global global_model
    if global_model is None:
        global_model = YOLO('../models/yoloNano.pt')
        if torch.cuda.is_available():
            global_model.to('cuda')
    return global_model

def run_yolo(frame):
    global global_model
    
    if global_model is None:
        global_model = load_model()
        if torch.cuda.is_available():
            global_model.to('cuda')
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Run inference
    results = global_model.predict(image)
    
    # Check if model returns oriented bounding boxes (OBB) or regular boxes
    if hasattr(results[0], 'obb') and results[0].obb is not None:
        detections = results[0].obb.xywhr.cpu().numpy()  # Oriented bounding boxes
        classes = results[0].obb.cls.cpu().numpy()  # Class IDs
    else:
        detections = results[0].boxes.xywh.cpu().numpy()  # Regular bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

    noOfDetections = len(detections)

    # Draw bounding boxes on the frame
    for i in range(noOfDetections):
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            x, y, w, h, r = detections[i]
            rect = ((x, y), (w, h), np.degrees(r))
            box = cv2.boxPoints(rect)
            box = np.intp(box)
        else:
            x, y, w, h = detections[i]
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        cv2.polylines(frame, [box], True, (0,0,255), 3)
    
    cv2.putText(frame, str(noOfDetections), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)

    # Convert detections to standard boxes [x1, y1, x2, y2] and include class IDs
    pred_boxes = []
    pred_classes = []
    for det, cls in zip(detections, classes):
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            x, y, w, h, r = det
        else:
            x, y, w, h = det
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        pred_boxes.append([x1, y1, x2, y2])
        pred_classes.append(int(cls))

    return frame, pred_boxes, pred_classes