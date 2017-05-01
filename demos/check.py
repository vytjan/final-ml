import cv2
import numpy as np

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
print(responses)
responses = responses.reshape((responses.size,1))

print(responses)
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

kernel = np.ones((2,2),np.uint8)

img = cv2.imread('ja_test.png')
# img = cv2.resize(img, (640, 360))
newx,newy = img.shape[1]/3,img.shape[0]/3     #new size (w,h)
print("Rescaled, new dimensions: ", newx, newy)
newimage = cv2.resize(img,(int(newx), int(newy)))
out = newimage

img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# dilation = cv2.dilate(im_bw,kernel,iterations = 5)
# im_bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# dilation = cv2.dilate(im_bw,kernel,iterations = 1)
eroded = cv2.erode(im_bw,kernel,iterations = 1)
cv2.imshow("Image", eroded)
cv2.waitKey(0)
im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
index = 0
# list of bounding rects
bounding = []
# classification
samples =  np.empty((0,100))
responses = []


for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,255,0),2)
            roi = eroded[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.imshow('im',eroded)
cv2.imshow('out',out)
cv2.waitKey(0)