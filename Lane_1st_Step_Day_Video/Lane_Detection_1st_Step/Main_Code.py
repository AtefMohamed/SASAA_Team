#######################################################
# Liberaries
import numpy as np
import cv2
import pickle
import matplotlib as plt
from Lane_Detection_1st_Step import Initialization_1step as init

####################################################
# Initialization
videoPath = r'E:\4 Mechatronics\2nd Term\image\Project\Image_Proccessing.mp4'  # Video Path -Copy the directory of the target video-
intialTracbarVals = [42, 63, 14, 87]  # wT,hT,wB,hB         # for further editiing
cap = cv2.VideoCapture(videoPath)   # read the video and store it to cap
frameWidth= 480                     # to resize the frame
frameHeight = 320
count = 0
noOfArrayValues = 10
global arrayCurve, arrayCounter
arrayCounter = 0
arrayCurve = np.zeros([noOfArrayValues])
myVals = []

init.initializeTrackbars(intialTracbarVals)    # calling for

while True:
    success, img = cap.read()      # read frame of the video
    img = cv2.resize(img, (frameWidth, frameHeight), None)  # resize
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    imgUndis = init.undistort(img)     # undistored the video to enhance it's quality
    # cv2.imshow('tt',img)     # testing true image
    # cv2.imshow('un', imgUndis)   # testing undistorted image
    imgThres, imgCanny, imgColor = init.thresholding(imgUndis)
    #cv2.imshow('thres',imgThres)
    #cv2.imshow('canny',imgCanny)
    #cv2.imshow('true', img)
    #cv2.imshow('clor', imgColor)
    #src = init.valTrackbars()

    #masked = init.region_of_interest(img,int[190],170,120,180)
    #cv2.imshow('masked',masked)    # Maske the road ONLY

    #plt.subplot(), plt.imshow(img ,cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(), plt.imshow(imgCanny, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #fig = plt.figure(figsize=(8, 8))
    #plt.show()
    imgStacked = init.stackImages(0.7, ([img, imgUndis, imgWarpPoints],
                                         [imgColor, imgCanny, imgThres]

                                         ))
    cv2.imshow("PipeLine", imgStacked)   # show the output
    #cv2.imshow("Result", imgFinal)
    #cv2.imshow('ff',imgCanny)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #to stream the output video
        break

cap.release()
cv2.destroyAllWindows()