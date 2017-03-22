import cv2
import numpy as np
def captch_ex(file_name ):
    img  = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    # image_final = cv2.bitwise_and(img2gray , img2gray , mask =  mask)

    cv2.imshow('captcha_result' , img2gray)
    cv2.waitKey()
    ret,thresh = cv2.threshold(img2gray,90,255,cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
    index = 0
    # list of bounding rects
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
        cv2.rectangle(img2gray,(x,y),(x+w,y+h),(0,0,255),2)
        roi = thresh[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
        cv2.imshow('norm',img2gray)
        key = cv2.waitKey(0)

        if key == 27:  # (escape to quit)
            sys.exit()
        else:
            responses.append(int(chr(key)))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print ("training complete")

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
        # detect dots on i:
    # newbounding = []
    # for bunit in bounding:
    #     for nestedbunit in bounding:
    #         if nestedbunit[0] > bunit[0] and nestedbunit[0] < bunit[0]+bunit[2] and bunit[1] > nestedbunit[1]:
    #             # tempboundunit = [bunit[0], nestedbunit[1], bunit[2], bunit[3]]
    #             newbounding.append([bunit[0], nestedbunit[1], bunit[2], bunit[1]-nestedbunit[1]+bunit[3]])
    #             bounding.remove(nestedbunit)
    #             continue
    #         if (bunit[0] < nestedbunit[0] < (bunit[0]+bunit[2])) and (bunit[1] < nestedbunit[1] < (bunit[1] + bunit[3])):
    #             bounding.remove(nestedbunit)
    #             continue
    #         # else:
    #             # newbounding.append([nestedbunit[0], nestedbunit[1], nestedbunit[2], nestedbunit[3]])
    # for rect in newbounding:
    #     if rect[0] > 10 and rect[1] > 10:
    #         cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(255,0,255),2)
            # try to classify right here:

        #you can crop image and send to OCR  , false detected will return no text :)
        # cropped = img2gray[rect[1] : rect[1]+rect[3] , rect[0] : rect[0]+rect[2]]

        # s = 'j_' + str(index) + '.png' 
        # cv2.imwrite(s, cropped)
        # index = index + 1
    # need to get 
    # srtdymax = sorted(bounding, key = lambda x: x[1]+x[3])
    # srtdymin = sorted(bounding, key = lambda x: x[1])
    # srtdxmin = sorted(bounding, key = lambda x: x[0])
    # srtdxmax = sorted(bounding, key = lambda x: x[0]+x[2])
    # print(srtdxmax[-1][0]+srtdxmax[-1][2], srtdxmin[0][0])
    # print(srtdymin[0][1], srtdymax[-1][1]+srtdymax[-1][3])
    # cv2.rectangle(img,(srtdxmin[0][0],srtdymin[0][1]),(srtdxmax[-1][0]+srtdxmax[-1][2],srtdymax[-1][1]+srtdymax[-1][3]),(255,0,255),2)
    # # write original image with added contours to disk  
    # cv2.imshow('captcha_result' , img2gray)
    # cv2.waitKey()


file_name ='Alearn.png'
# file_name ='4.jpg'
captch_ex(file_name)