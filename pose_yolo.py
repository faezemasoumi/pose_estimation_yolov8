#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:15:37 2024

@author: mehdi
"""

import cv2
from ultralytics import YOLO
import numpy as np


# Load a model
model = YOLO("yolov8s-pose.pt")  # load an official model

video_path = 2
cap = cv2.VideoCapture(video_path)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
                
        #======= detection part using yolo ===========
        output = model(frame)

        #==== show detection boxes using yolos ========
        annotated_frame = output[0].plot()
        cv2.imshow('output', annotated_frame)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
      
        


