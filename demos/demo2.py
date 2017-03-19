import cv2
import numpy as np
def captch_ex(file_name ):
    img  = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray , img2gray , mask =  mask)

    ret,thresh = cv2.threshold(img2gray,127,255,cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
    index = 0
    # list of bounding rects
    bounding = []
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        bounding.append([x,y,w,h])
        # print(y, (y+(y+h))/2)
        #Don't plot small false positives that aren't text
        if w < 15 and h<15:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    srtdy = sorted(bounding, key = lambda x: x[1])
    srtdx = sorted(bounding, key = lambda x: x[0])
    print(srtdx[-1][0]+srtdx[-1][2], srtdx[0][0])
    print(srtdy[0][1], srtdy[-1][1]+srtdy[-1][3])
    cv2.rectangle(img,(srtdx[0][0],srtdy[0][1]),(srtdx[-1][0]+srtdx[-1][2],srtdy[-1][1]+srtdy[-1][3]),(255,0,255),2)
    # write original image with added contours to disk  
    cv2.imshow('captcha_result' , img)
    cv2.waitKey()


file_name ='sample_text.png'
# file_name ='4.jpg'
captch_ex(file_name)