import cv2
import numpy as np

def testing():

    ############################# testing part  #########################

    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('train_2_words.png')
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
    eroded = cv2.erode(im_bw,kernel,iterations = 3)
    # cv2.imshow("Image", eroded)
    # cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []
    # # classification  
    # samples =  np.empty((0,100))
    # responses = []

    contours.pop(0)

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if w < 20 and h<20:
            continue
        bounding.append([x,y,w,h])

    newbounding = removeInnerContours(eroded, bounding, im_bw)
    maxHeight = newbounding[0][3]
    for a in newbounding:
        if a[3] > maxHeight:
            maxHeight = a[3]
    print("max height is: ", maxHeight)

    for single in newbounding:
        [x,y,w,h] = [single[0], single[1], single[2], single[3]]
        # print([x,y,w,h])
        adjustWidth([x,y,w,h], eroded, maxHeight, out)
        # cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)
    #     roi = eroded[y:y+h,x:x+w]
    #     roismall = cv2.resize(roi,(10,10))
    #     roismall = roismall.reshape((1,100))
    #     roismall = np.float32(roismall)
    #     retval, results, neigh_resp, dists = model.findNearest(roismall, k = 2)
    #     print("retrieved value is: ", retval)
    #     print("results are: ", results)
    #     print("Neighbor responses are: ", neigh_resp)
    #     print("Distances from input vectors: ", dists)
    #     string = str(int((results[0][0])))
    #     cv2.putText(out,string,(x,y+h),0,1,(255,0,0))

    # cv2.imshow('im',eroded)
    # cv2.imshow('out',out)
    # cv2.waitKey(0)

# extract the words into letters:
def adjustWidth(coords, eroded, maxHeight, out):
    # global samples

    #######   training part    ###############
    samples = np.loadtxt('generalsamplesb.data',np.float32)
    responses = np.loadtxt('generalresponsesb.data',np.float32)
    print(responses)
    responses = responses.reshape((responses.size,1))

    print(responses)
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    [x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]

    contourWidth = x + w
    sampleWidth = 14
    # positions: a j N J. Default - a
    heightPos = 1
    y = int(y - h/2)
    h = int(1.5*h)
    
    while x+sampleWidth < contourWidth + 5 and x >= 0 and y >=0:
        # here we should find the nearest neighbours, thickering the width +2px every iteration:
        # cropToLetter = eroded[y:y+h, x:x+sampleWidth]
        # print(x+sampleWidth)
        # cv2.imshow("cropped", cropToLetter)
        # key = cv2.waitKey(0)
        # declare empty array:
        lettersWidth = []

        while sampleWidth < 40:

            # test every 2 px, store values and then decide which one to choose:
            cropToLetter  = eroded[y:y+h, x:x+sampleWidth]

            # print(x+sampleWidth)
            roi = eroded[y:y+h, x:x+sampleWidth]
            roismall = cv2.resize(roi,(10,30))
            roismall = roismall.reshape((1,300))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            lettersWidth.append([dists[0][0], sampleWidth, sampleHeight, results[0][0]])
            # print("retrieved value is: ", retval)
            # print("results are: ", results[0][0])
            # print("Neighbor responses are: ", neigh_resp)
            # print("Distances from input vectors: ", dists)
            # string = chr(int(results[0][0]))
            # cv2.imshow("cropped", cropToLetter)
            # key = cv2.waitKey(0)
            sampleWidth = sampleWidth + 2

        # here I decide which is the best 'guess':
        bestGuess = sortGuesses(lettersWidth)

        string = chr(int(bestGuess[2]))
        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
        cv2.rectangle(eroded,(x,y),(x+bestGuess[1],y+h),(0,0,0),2)
        # reset start x value:
        x = x + bestGuess[1]

        sampleWidth = 14
        # print(lettersWidth)
        continue
            # if key == 27:
            #     sys.exit()
            # elif key == 83:
            #     sampleWidth += 2
            #     continue
            # # up arrow
            # elif key == 82:
            #     y = y - 0.5*h
            #     if(y < 0):
            #         y = 0
            #     h = 1.5*h
            #     continue
            # # down arrow
            # elif key == 84:
            #     # y = y + (maxHeight - h)
            #     h = maxHeight
            #     continue
            # # go to previous width:
            # elif key == 81:
            #     sampleWidth -= 2
            #     continue
            # # if a letter:
            # elif key == 32:
            #     y = y - h
            #     if(y < 0):
            #         y = 0
            #     h = 1.5*h
            #     continue
            # else:
            #     if key == 226:
            #         key = chr(cv2.waitKey(0))
            #         key = ord(key.upper())
            #     else:
            #     # s = input("Write a capital letter ")
            #     # print(s)
            #     # submitKey = key
            #     # print(chr(key))
            #         print(key)
            #     print(key)
            #     roi = eroded[y:y+h,x:x+w]
            #     roismall = cv2.resize(roi,(10,10))
            #     responses.append(int(key))
            #     sample = roismall.reshape((1,100))
            #     samples = np.append(samples,sample,0)
            #     cv2.imshow('Added to classification',cropToLetter)
            #     cv2.waitKey(0)
            #     # reset start x value:
            #     x = x + sampleWidth
            #     sampleWidth = 20
            #     continue

        #     roi = eroded[y:y+h, x:x+sampleWidth]
        #     roismall = cv2.resize(roi,(10,10))
        #     roismall = roismall.reshape((1,100))
        #     roismall = np.float32(roismall)
        #     retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        #     # print("retrieved value is: ", retval)
        #     print("results are: ", results[0][0])
        #     # print("Neighbor responses are: ", neigh_resp)
        #     print("Distances from input vectors: ", dists)
        #     string = chr(int(results[0][0]))
        # cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
        # cv2.rectangle(eroded,(x,y),(x+sampleWidth,y+h),(0,0,0),2)
        # reset start x value:

    cv2.imshow('im',eroded)
    cv2.imshow('out',out)
    cv2.waitKey(0)

def sortGuesses(letters):
    closest = letters[0]
    for single in letters:
        if single[0] < closest[0]:
            closest = single
    print("Closest guess is: ", closest)
    return closest  


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
        if (bunit[2] > 15 and bunit[3] > 15):
            newbounding.append([bunit[0], bunit[1], bunit[2], bunit[3]])
            continue
    print(len(newbounding))

    return newbounding

testing()