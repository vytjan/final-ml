import cv2
import numpy as np
import difflib
import codecs

#######   training part    ###############
global samples
global responses
global dictionary
# samples = np.loadtxt('generalsamplesA.data',np.float32)
# responses = np.loadtxt('generalresponsesA.data',np.float32)

# dictionary = np.loadtxt('dictionary.dat', delimiter="\n")
# dictionary = np.loadtxt('dictionary.dat', converters={0: lambda x: unicode(x, 'utf-8')}, dtype='U2', delimiter="\n")
dictionary = codecs.open("dictionary.dat", encoding="utf-8").read()
# s = codecs.open("dictionary.dat", encoding="utf-8")
# arr = numpy.frombuffer(s.replace("\n", ""), dtype="<U2")
# print(arr)
# dictionary = np.loadtxt('dictionary.dat', delimiter="\n")
print(dictionary)

something = difflib.get_close_matches('yna', dictionary)
print("zodis yra ",something)

samples = np.loadtxt('generalsamplesA.data',np.float32)
responses = np.loadtxt('generalresponsesA.data',np.float32)
# print(responses)
responses = responses.reshape((responses.size,1))

# print(responses)
global model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

def testing():

    ############################# testing part  #########################

    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('gb3.png')
    # img = cv2.resize(img, (640, 360))
    newx,newy = img.shape[1]/4,img.shape[0]/4    #new size (w,h)
    print("Rescaled, new dimensions: ", newx, newy)
    newimage = cv2.resize(img,(int(newx), int(newy)))
    out = newimage

    img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
    # (thresh, thresh1) = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    # binary threshold works for the greenboard.
    (thresh, thresh1) = cv2.threshold(img, 110,255,cv2.THRESH_BINARY)
    # cv2.imshow("inverted", thresh1)
    # cv2.waitKey(0)

    # thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            # cv2.THRESH_BINARY,15,11)

    im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("contours length",len(contours))
    # remove the first contour:
    contours.pop(0)
    # find the largest contour:
    c = max(contours, key = cv2.contourArea)
    contours.pop(0)
    [x,y,w,h] = cv2.boundingRect(c)
    out = newimage[y:y+h, x:x+w]
    kernel2 = np.ones((2,2),np.uint8)
    # dilation = cv2.dilate(thresh1,kernel,iterations = 1)
    # cv2.imshow("dilated", dilation)
    # cv2.waitKey(0)
    eroded = cv2.dilate(thresh1,kernel2,iterations = 1)
    newImage = eroded[y:y+h, x:x+w]
    # cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,0,0),2)
    # (thresh, thresh1) = cv2.threshold(newImage, 127,255,cv2.THRESH_BINARY_INV)
    # th = cv2.bitwise_not(newImage)
    th = cv2.bitwise_not(newImage)
    thresh1 = cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    	cv2.THRESH_BINARY,11,11)

    # cv2.imshow("inverted new image", thresh1)
    # cv2.waitKey(0)
    # cv2.imshow("contour", newImage)
    # cv2.waitKey(0)  

    # for cnt in contours:
    #     [x,y,w,h] = cv2.boundingRect(cnt)
    #     if (w < 20 and h<20) or (h > 180):
    #         continue
    #     # bounding.append([x,y,w,h])
    # cv2.rectangle(thresh1,(x,y),(x+w,y+h),(0,0,0),2)

    # cv2.imshow("find contours", thresh1)
    # cv2.waitKey(0)
    # blur = cv2.GaussianBlur(img,(4,4),0)
    # ret3,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    # cv2.imshow("threshold", thresh1)
    # cv2.waitKey(0)

    kernel2 = np.ones((2,2),np.uint8)
    # dilation = cv2.dilate(thresh1,kernel,iterations = 1)
    # cv2.imshow("dilated", dilation)
    # cv2.waitKey(0)
    eroded = cv2.erode(thresh1,kernel2,iterations = 1)
    # eroded = cv2.dilate(eroded,kernel,iterations = 1)
    cv2.imshow("eroded", eroded)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []
    # # classification  
    # samples =  np.empty((0,100))
	# responses = []

	# contours.pop(0)
    # [x,y,w,h] = cv2.boundingRect(contours[0])
    # newImage = eroded[y:y+h, x:x+w]
    # cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,0,0),2)
    # cv2.imshow("new image", newImage)
    # cv2.waitKey(0)    

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (w < 20 and h<20) or (h > 180):
            continue
        bounding.append([x,y,w,h])
        # cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,0),2)

    cv2.imshow("contours", eroded)
    cv2.waitKey(0)

    newbounding = removeInnerContours(eroded, bounding, newImage)

    maxHeight = newbounding[0][3]
    for a in newbounding:
        if a[3] > maxHeight:
            maxHeight = a[3]
    # print("max height is: ", maxHeight)

    for single in newbounding:
        [x,y,w,h] = [single[0], single[1], single[2], single[3]]
        # print([x,y,w,h])
        adjustWidth([x,y,w,h], eroded, maxHeight, out, newx, newy)
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
def adjustWidth(coords, eroded, maxHeight, out, newx, newy):
    textValue = []
    [x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]

    contourWidth = x + w
    sampleWidth = 13
    # positions: a j N J. Default - a
    # print("max height is: ", maxHeight)
    print("new contour is iterated:          ")    
    while x+sampleWidth < contourWidth and x >= 0 and y >=0:
        # here we should find the nearest neighbours, thickering the width +2px every iteration:
        # cropToLetter = eroded[y:y+h, x:x+sampleWidth]
        # print(x+sampleWidth)
        # cv2.imshow("cropped", cropToLetter)
        # key = cv2.waitKey(0)
        # declare empty array:
        lettersWidth = []
        while sampleWidth < 40 and x + sampleWidth < contourWidth:

            # get results of the height variations of the letters:
            heightVariations = []

            # clean up this stuff later:------------------------------------------------------------------
            # If J, do nothing:
            heightVariations.append([y,h])
            # if l, increase below:
            # if h*1.5 < 85:
            height = h
            while height < 70:
                heightVariations.append([y,height])
                height += 4
                if y-h/2 > 0 and h < 50:
                    heightVariations.append([y-h/2, height])
                elif y-h > 0 and h < 50:
                    heightVariations.append([y-h, height])
                # height += 5
                # heightVariations.append([y,h*1.1])
            # if j:
            # if y-h/2 > 0 and h*1.5 < 1.1*maxHeight:
            # if y-h/2 > 0 and h*1.5 < 85:
            # 	height = h
            # 	while height < 85:

            #         heightVariations.append([y-h/2, height])
            #         height += 5
                # heightVariations.append([y-h/2, h*1.2])
            # if a: 
            # if y-h > 0 and h*3 < 1.1*maxHeight:
            # if y-h > 0 and h*3 < 85:
            # 	height = h
            # 	while height < 85:

            #         heightVariations.append([y-h, height])
            #         height += 5
                # heightVariations.append([y-h, h*1.5])

            # print("length of heightvars: ", len(heightVariations))
            for singleCoord in heightVariations:
                # print("new iteration of getting roi: \n")
                xplus = 0
                while xplus < 5:
                    results, dists = getRoi(eroded, [x + xplus, singleCoord[0], sampleWidth, singleCoord[1]])
                    lettersWidth.append([dists, x + xplus, singleCoord[0], sampleWidth, singleCoord[1], results[0][0]])
                    xplus += 1
            # Get needed roi:
        
            # print("retrieved value is: ", retval)
            # print("results are: ", results[0][0])
            # print("Neighbor responses are: ", neigh_resp)
            # print("Distances from input vectors: ", dists)
            # string = chr(int(results[0][0]))
            # cv2.imshow("cropped", cropToLetter)
            # key = cv2.waitKey(0)
            sampleWidth = sampleWidth + 1

        # here I decide which is the best 'guess':
        # print("length is: ", len(lettersWidth))
        if len(lettersWidth) > 0:
            bestGuess = sortGuesses(lettersWidth)
        else:
            break

        string = chr(int(bestGuess[5]))
        textValue.append(string)
        cv2.putText(out,string,(int(bestGuess[1]), int(bestGuess[2]+bestGuess[4])),0,1,(0,0,255))
        cv2.rectangle(out,(int(bestGuess[1]),int(bestGuess[2])),(int(x+bestGuess[3]), int(bestGuess[2]+bestGuess[4])),(0,0,0),1)
        # reset start x value:
        x = x + bestGuess[3] -1

        sampleWidth = 13
        # print(lettersWidth)
        continue

    print(textValue)
    cv2.imshow('im',eroded)
    cv2.imshow('out',out)
    cv2.waitKey(0)

def sortGuesses(letters):
    closest = letters[0]
    for single in letters:
    	# print("sum of the closest is:", np.sum(single[0]))
    	if np.sum(single[0]) < np.sum(closest[0]):
    		closest = single
    	# print("single is: ",single)
    	# if single[0] < closest[0]:
        	# closest = single
    # print("sum of the closest is:", np.sum(closest[0]))
    return closest  



# get the guess from the rate of interests:
def getRoi(eroded, coords):
    global model
    x,y,w,h = [coords[0], coords[1], coords[2], coords[3]]
    roi = eroded[y:y+h, x:x+w]
    roismall = cv2.resize(roi,(10,30))
    # cv2.imshow("trial", roi)
    # cv2.imshow("roismall", roismall)
    # cv2.waitKey(0)
    roismall = roismall.reshape((1,300))
    roismall = np.float32(roismall)
    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 5)
    # print("distances are; ", dists)
    # print("results are: ", chr(int(results[0][0])))
    # print("neighour response is: ",neigh_resp)
    # print("ret value is: ", retval)
    return results, dists


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
    #   print(len(newbounding))

    return newbounding

testing()