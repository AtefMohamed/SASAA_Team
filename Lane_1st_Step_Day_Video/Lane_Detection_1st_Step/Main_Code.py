#######################################################
# Liberaries
import numpy as np
import cv2
import pickle
import matplotlib.pylab as plt
from Lane_Detection_1st_Step import Initialization_1step as init
#from Lane_Detection_1st_Step import init_new as initn
import imghdr


####################################################
# Initialization
videoPath = r'E:\4 Mechatronics\2nd Term\image\Project\Image_Proccessing.mp4'  # Video Path -Copy the directory of the target video-
intialTracbarVals = [42, 63, 19, 87]  # wT,hT,wB,hB         # for further editiing
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
car_cascade = cv2.CascadeClassifier('Cars3.xml')
while True:
    success, img = cap.read()      # read frame of the video
    img = cv2.resize(img, (frameWidth, frameHeight), None)  # resize
    warped_img_Points = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    imgCar = img.copy()
    gray = cv2.cvtColor(imgCar, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    # To draw a rectangle in each cars



    #imgUndis = init.undistort(img)     # undistored the video to enhance it's quality
    # cv2.imshow('tt',img)     # testing true image
    # cv2.imshow('un', imgUndis)   # testing undistorted image
    imgThres, imgCanny, imgColor = init.thresholding(img)
    #cv2.imshow('thres',imgThres)
    #cv2.imshow('canny',imgCanny)
    #cv2.imshow('true', img)
    #cv2.imshow('clor', imgColor)
    #src = init.valTrackbars()



    #plt.subplot(), plt.imshow(img ,cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(), plt.imshow(imgCanny, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #fig = plt.figure(figsize=(8, 8))
    #plt.show()

    src=init.valTrackbars()
    imgWarp = init.warp_img(imgThres, destination_points_size=(frameWidth, frameHeight), src=src)
    warped_img_Points = init.draw_Points(warped_img_Points, src)
    imgSliding, curves, lanes, ploty = init.sliding_window(imgWarp, draw_windows=True)
    ##############################################################################
              #            Test Code
    ROI = [(90,270),(195,195),(270,200),(360,280)]
    cropped_img = init.region_of_interest_new(imgThres,np.array([ROI], np.int32), )
    #print('width ' , img.shape[1])
    #cv2.imshow("ti",cropped_img)

    lines = cv2.HoughLinesP(cropped_img,rho=2,theta=np.pi/180,threshold=50,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=250)
    image_with_lines ,line_n = init.draw_line_new(img,lines)
    #cv2.imshow("ti",image_with_lines)
    #cv2.imshow('crp',cropped_img)
    #cv2.imshow('crp', line_n)
    #plt.imshow(image_with_lines)
    #plt.show()
    #img = cv2.imread('test_images/test3.jpg')
    """""""""
    img_n = cv2.cvtColor(imgUndis, cv2.COLOR_BGR2RGB)
    dst = initn.pipeline(img_n)
    dst = initn.perspective_warp(dst, dst_size=(1280, 720))

    # Visualize undistortion
    cv2.imshow('im',img)
    cv2.imshow('dst',dst)
    #cv2.imshow('dst',camp)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    
    """
    ############################################################################

    try:
        curverad = init.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        #print('curve',lane_curve)
        imgFinal = init.draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

        # ## Average
        currentCurve = lane_curve // 50
        if int(np.sum(arrayCurve)) == 0:
            averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve - currentCurve) > 200:
            arrayCurve[arrayCounter] = averageCurve
        else:
            arrayCurve[arrayCounter] = currentCurve
        arrayCounter += 1
        if arrayCounter >= noOfArrayValues: arrayCounter = 0
        #cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth // 2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75,
                    #(0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgFinal, 'Lane Curvature: {:.0f} m'.format(averageCurve),
          (frameWidth // 2 - 210, 20),
                    cv2.FONT_ITALIC, 0.6,
                    (0, 0, 255),2, cv2.LINE_4)
        cv2.putText(imgFinal, 'Vehicle offset: {:.4f} m'.format(curverad[2]),
                    (frameWidth // 2 - 210, 50),
                    cv2.FONT_ITALIC, 0.6,
                    (0, 0, 255), 2, cv2.LINE_4)
        cv2.putText(imgFinal, 'SASAA Team ',
                    (310, 25),
                    cv2.FONT_ITALIC, 0.6,
                    (88, 219, 255), 2, cv2.LINE_4)

    except:
        lane_curve = 00
        pass
        
    ##########################TEXT#####################################################3
    imgFinal = cv2.addWeighted(line_n, 0.9, imgFinal, 1, 0.0)
    for (x, y, w, h) in cars:
        cv2.rectangle(imgFinal, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Display frames in a window
    #cv2.imshow('video2', imgFinal)

    masked_im = init.region_of_interest(img, 190, 170, 120, 180)
    cv2.putText(img, 'Orignal Frame ',(310, 25),cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_4)
    #cv2.putText(img, 'Unditorted frame ', (310, 25), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(warped_img_Points, 'Processed Area ', (310, 25), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(imgColor, 'Color Filtering ', (310, 25), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(imgCanny, 'Canny Filter ', (310, 25), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2, cv2.LINE_4)
    cv2.putText(imgThres, 'Threshold frame ', (310, 25), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_4)

    #imgFinal = init.drawLines(imgFinal, lane_curve)
    #imgFinal = cv2.addWeighted(line_n,0.8,imgFinal,1,0.0)
    #cv2.imshow('final', imgFinal)
    #imgThres = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR)
    #imgBlank = np.zeros_like(img)
    imgStacked = init.stackImages(0.7, ([img, img, warped_img_Points],
                                         [imgColor, imgCanny, imgThres]
                                        ,[imgWarp,imgSliding,imgFinal]))

    masked_im = init.region_of_interest(img, 190, 170, 120, 180)
    #cv2.imshow("masked_im", masked_im)  # Maske the road ONLY
    cv2.imshow("PipeLine", imgStacked)   # show the output
    #cv2.imshow("Result", imgFinal)
    ## cv2.imshow('ff',imgCanny)

    #plt.imshow(imgFinal)
    #plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):  #to stream the output video
        break

    #if cv2.getWindowProperty('Pipeline',4) < 1:
       #break

cap.release()
cv2.destroyAllWindows()
