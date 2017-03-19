# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

img = cv2.imread('4.jpg')
#img = cv2.resize(img, (480, 640))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#img = cv2.medianBlur(img,5)
#thresh  = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#cv2.imshow("Img", thresh)
#cv2.waitKey(0)
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #box = np.int0(box)
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    #    cv2.drawContours(im2, cnt, 0, (0,255,0), 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
