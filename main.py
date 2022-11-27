import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tracker import *

# Create tracker object for counting
tracker = EuclideanDistTracker()


### Object detection from Stable camera
## this func will extract the moving obj from the stable video
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)    ## varThreshold is lower false +ve will increase

cap =  cv2.VideoCapture("highway.mp4")

while True:
    ret, frame = cap.read()

    print(frame.shape)                    ## (720, 1280, 3)


    ### Region of interest (ROI)
    roi = frame[340:720, 500:800]           ## can find using FindROI


    ### applying the object_detector on roi
    mask = object_detector.apply(roi)

    ### 1. Object Detection & Tracking using contours
    ## for contours we need threshold frame to remove the shadow (grey noise), keeping only white pixles
    ret, thres = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)    
    contours, hierarchy  = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    ### NOTE: using [-2:]  for ValueError: too many values to unpack (expected 2)


    detections = []
    for contour in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(contour)
        
        if area > 500:
            #cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)       ## on the place of drawContours we drawing boundingRect

            detections.append([x, y, w, h])       ### append to count the detected boundingRect


    ### 2. Object Countng
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.pyrDown(frame)
    cv2.imshow("roi", roi)
    cv2.imshow("Mask Video", mask)
    cv2.imshow("Video", frame)
    cv2.pyrDown(frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



