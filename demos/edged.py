# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

kernel = np.ones((2,2),np.uint8)
img = cv2.imread('ja2.png')
# img = cv2.resize(img, (640, 360))
newx,newy = img.shape[1]/3,img.shape[0]/3	  #new size (w,h)
print("Rescaled, new dimensions: ", newx, newy)
newimage = cv2.resize(img,(int(newx), int(newy)))

img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# dilation = cv2.dilate(im_bw,kernel,iterations = 5)
# im_bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# dilation = cv2.dilate(im_bw,kernel,iterations = 1)
erosion = cv2.erode(im_bw,kernel,iterations = 1)
cv2.imshow("Image", erosion)
cv2.waitKey(0)
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


bounding = []
# classification
samples =  np.empty((0,100))
responses = []

for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)
    bounding.append([x,y,w,h])
    print([x,y,w,h])
    #Don't plot small false positives that aren't text
    if w < 40 and h<40:
        continue
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    roi = thresh[y:y+h,x:x+w]
    roismall = cv2.resize(roi,(10,10))
    cv2.imshow('norm',img)
    key = cv2.waitKey(0)

    if key == 27:  # (escape to quit)
        sys.exit()


# for cnt in contours:
# 	rect = cv2.minAreaRect(cnt)
# 	box = cv2.boxPoints(rect)
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# 	box = np.int0(box)
# 	cv2.drawContours(img,[box],0,(0,0,255),2)
# 	cv2.drawContours(im2, cnt, 0, (0,255,0), 3)



cv2.imshow("Image", img)
cv2.waitKey(0)
