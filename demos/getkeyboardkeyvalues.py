import cv2
import time
img = cv2.imread('ja.png') # load a dummy image
while(1):
    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==255:  # normally -1 returned,so don't print it
        continue
    else:
        # millis = int(round(time.time() * 1000))
        # print(millis)
        print(str(round(time.time())))
        print(k) # else print its value