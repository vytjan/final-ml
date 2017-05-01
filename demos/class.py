import cv2
import numpy as np
def captch_ex(file_name ):
    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('ja_training.png')
    # img = cv2.resize(img, (640, 360))
    newx,newy = img.shape[1]/3,img.shape[0]/3     #new size (w,h)
    print("Rescaled, new dimensions: ", newx, newy)
    newimage = cv2.resize(img,(int(newx), int(newy)))

    img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # dilation = cv2.dilate(im_bw,kernel,iterations = 5)
    # im_bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # dilation = cv2.dilate(im_bw,kernel,iterations = 1)
    eroded = cv2.erode(im_bw,kernel,iterations = 3)
    cv2.imshow("Image", eroded)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []
    # classification
    samples =  np.empty((0,100))
    responses = []

    # remove main contour:
    contours.pop(0)

    # removal of contours
    # correct = []

    # # print(len(contours))
    # # remove inner contours:
    # for single in hierarchy[0]:
    #     print(single[2])
    #     if single[2] != -1:
    #         print(np.where(hierarchy[0] == single))
    #         contours.pop(contours[np.where(hierarchy[0] == single)])

    # for a in correct:
    #     print(a)
    # for contour in contours:
    #     # get rectangle bounding contour
    #     # if()
    #     [x,y,w,h] = cv2.boundingRect(contour)
    #     bounding.append([x,y,w,h])
    #     print([x,y,w,h])
    #     #Don't plot small false positives that aren't text
    #     if w < 10 and h<10:
    #         continue
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if w < 20 and h<20:
            continue
        bounding.append([x,y,w,h])

    newbounding = removeInnerContours(eroded, bounding, im_bw)

    for single in newbounding:
        [x,y,w,h] = [single[0], single[1], single[2], single[3]]
        print([x,y,w,h])
        cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)


        roi = eroded[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
        cv2.imshow('norm',eroded)
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
    # removeInnerContours(eroded, bounding, im_bw)
    # cv2.imshow('norm',eroded)
    # key = cv2.waitKey(0)

def removeInnerContours(eroded, bounding, im_bw):
    # remove contours from inside of other contours:
    # boundingUnit = [x,y, height, width]
    newbounding = []
    bounding2 = list(bounding)
    for bunit in bounding:
        for nunit in bounding2:
            if (nunit[0] > bunit[0] and nunit[0]+nunit[2] < bunit[0]+bunit[2] and nunit[1] > bunit[1] and nunit[1]+nunit[3] < bunit[1]+bunit[3]):            
                # newbounding.append([bunit[0], bunit[1], bunit[2], bunit[3]])
                bounding.remove(nunit)
                continue
            # if nunit[0] > bunit[0] and nunit[0] < bunit[0]+bunit[2] and bunit[1] > nunit[1]:
            #     # tempboundunit = [bunit[0], nunit[1], bunit[2], bunit[3]]
            #     newbounding.append([bunit[0], nunit[1], bunit[2], bunit[1]-nunit[1]+bunit[3]])
            #     bounding.remove(nunit)
            #     continue
            # if (bunit[0] < nunit[0] < (bunit[0]+bunit[2])) and (bunit[1] < nunit[1] < (bunit[1] + bunit[3])):
            #     bounding.remove(nunit)
            # else:
            #     newbounding.append([nunit[0], nunit[1], nunit[2], nunit[3]])
        if (bunit[2] > 15 and bunit[3] > 15):
            newbounding.append([bunit[0], bunit[1], bunit[2], bunit[3]])
            continue
    print(len(newbounding))
            # bounding.remove(bunit)
    # for rect in newbounding:
    #     # if rect[0] > 10 and rect[1] > 10:
    #     cv2.rectangle(eroded,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(90,90,90),2)
        #     try to classify right here:

        # you can crop image and send to OCR  , false detected will return no text :)
        # cropped = img[rect[1] : rect[1]+rect[3] , rect[0] : rect[0]+rect[2]]

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
    # cv2.rectangle(eroded,(srtdxmin[0][0],srtdymin[0][1]),(srtdxmax[-1][0]+srtdxmax[-1][2],srtdymax[-1][1]+srtdymax[-1][3]),(0,0,0),2)
    # write original image with added contours to disk  
    # cv2.imshow('captcha_result' , eroded)
    # cv2.imshow('original contours' , im_bw)
    # cv2.waitKey()
    return newbounding

file_name ='ja_train.png'
# file_name ='4.jpg'
captch_ex(file_name)