#Not complete
import cv2
import numpy as np
import time

capture_video = cv2.VideoCapture("video.mp4")

# give the camera to warm up
time.sleep(3)
count = 0
background = 0
#capture background in range of 60 so it has enough time to process the background clearly
for i in range(60):
    return_val, background = capture_video.read()
    if return_val == False:
        continue
#flipping the frame         
background = np.flip(background, axis=1) 

    #converting from RGB to HSV 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ranges should be carefully chosen
    # setting the lower and upper range for mask1
    lower_red = np.array([100, 40, 40])
    upper_red = np.array([100, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # setting the lower and upper range for mask2
    lower_red = np.array([155, 40, 40])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
