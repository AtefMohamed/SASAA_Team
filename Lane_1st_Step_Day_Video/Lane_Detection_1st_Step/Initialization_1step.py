#######################################################
#Liberaries
import numpy as np
import cv2
import matplotlib.pylab as plt
import pickle
####################################################

def nothing(x):  # used in trackbar function to put NULL in the "onChange Argument"
    pass


def initializeTrackbars(intialTracbarVals):  # Creates a trackbar and attaches it to the specified window.
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], 50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, nothing)

"""
#path ='\Screenshot_1.png'
img1 = cv2.imread( r'Lane_Detection_1st_Step\Lane_Detection_1st_Step\Screenshot_1.png')
cv2.imshow('image1',img1)
plt.imshow(img1)
cv2.waitKey(0)
"""


# cal_pickle include the caliberation matrix and other variables to use it in undistorting frames
# Pickle is Object Serialization module to serialize the 5 parameters of the camera
def undistort(img, cal_dir='cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def colorFilter(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #lowerYellow = np.array([18,94,140])
    #upperYellow = np.array([48,255,255])
    lowerWhite = np.array([0, 0, 180])
    upperWhite = np.array([255, 255, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    #maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    #combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow)
    return maskedWhite

# thresholding the frames
def thresholding(img):
    imgGray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert the image ro gray scale
    kernel      = np.ones((5,5))                          #  Return a new array of given shape with ones.
    imgBlur     = cv2.GaussianBlur(imgGray, (5, 5), 0)    # Gaussian blue used to reduce noise, it's Unsharp masking to use in edge detection
    imgCanny    = cv2.Canny(imgBlur, 50, 100)             # canny algorithm used for edge detection
    #imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    imgDial     = cv2.dilate(imgCanny,kernel,iterations=1)  # erosion remove white noise and shrink the objects so we dilate it
    imgErode    = cv2.erode(imgDial,kernel,iterations=1)   # erosion remove noise and detach two connected components
    imgColor    = colorFilter(img)
    combinedImage = cv2.bitwise_or(imgColor, imgErode)

    return combinedImage,imgCanny,imgColor

def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                      (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    return src


def region_of_interest(img,a,b,c,d):
    height = img.shape[0]
    cv2.imshow('maske', masked)

    # Crop image

    imCrop = img[a:b + 170, c:d + 150]
    cv2.imshow('Image66', imCrop)
    # cv2.selectROI('ROI', imCrop)
    return masked_image

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

