import cv2
import numpy as np
import difflib
import codecs


global samples
global responses
global dictionary
global metadata

# load the dictionary:
dictionary = codecs.open("dictionary.dat", encoding="utf-8").read().splitlines()
metadata = codecs.open("lettertags.txt", encoding="utf-8").read().splitlines()
# print(dictionary)

samples = np.loadtxt('generalsamplesNEW.data',np.float32)
responses = np.loadtxt('generalresponsesNEW.data',np.float32)
responses = responses.reshape((responses.size,1))

global model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

def testing():

    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('gb13.png')
    newx,newy = img.shape[1]/3.5,img.shape[0]/3.5    #resize (w,h)
    print("New dimensions are: ", newx, newy)
    newimage = cv2.resize(img,(int(newx), int(newy)))
    # out = newimage
    # cv2.imshow("color image", out)
    # cv2.waitKey(0)

    img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)	

    (thresh, thresh1) = cv2.threshold(img, 127,255,cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("contours length", len(contours))
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
    eroded = cv2.dilate(thresh1,kernel2,iterations = 2)

    newImage = eroded[y:y+h, x:x+w]
    th = cv2.bitwise_not(newImage)
    thresh1 = cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    	cv2.THRESH_BINARY,43,43)

    kernel2 = np.ones((2,2),np.uint8)
    eroded = cv2.dilate(thresh1,kernel,iterations = 1)
    eroded = cv2.erode(thresh1,kernel2,iterations = 2)
    eroded = cv2.dilate(eroded,kernel,iterations = 1)
    cv2.imshow("eroded", eroded)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []

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
    print("max height is: ", maxHeight)

    for single in newbounding:
        [x,y,w,h] = [single[0], single[1], single[2], single[3]]
        adjustWidth([x,y,w,h], eroded, maxHeight, out, newx, newy)
        # cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)

    cv2.imshow('im',eroded)
    cv2.waitKey(0)
    cv2.imshow('out',out)
    cv2.waitKey(0)

# extract the words into letters:
def adjustWidth(coords, eroded, maxHeight, out, newx, newy):
    global dictionary
    textValue = []
    [x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]

    contourWidth = x + w
    sampleWidth = 13
    # positions: a j N J. Default - a
    # print("max height is: ", maxHeight)
    print("new contour is iterated:          ")    
    while x+sampleWidth < contourWidth and x >= 0 and y >=0:
        # here we should find the nearest neighbours, thickering the width +2px every iteration:
        # declare empty array:
        lettersWidth = []
        while sampleWidth < 48 and x + sampleWidth < contourWidth+8:

            # get results of the height variations of the letters:
            heightVariations = []

            # clean up this stuff later:------------------------------------------------------------------
            # If J, do nothing:
            heightVariations.append([y,h])
            # if l, increase below:
            # if h*1.5 < 85:
            height = h
            while height < 80:
                # print(height)
                heightVariations.append([y,height])
                # height += 5
                if y-height/2 > 0 and height < 80:
                    heightVariations.append([y-height/2, height])
                elif y-height > 0 and height < 80:
                    heightVariations.append([y-height, height])
                height += 1

            # print("length of heightvars: ", len(heightVariations))
            for singleCoord in heightVariations:
                # print("new iteration of getting roi: \n")
                xplus = 0
                # while xplus < 7:
                # print(x, singleCoord[0], sampleWidth, singleCoord[1])
                results, dists = getRoi(eroded, [x, singleCoord[0], sampleWidth, singleCoord[1]])
                lettersWidth.append([dists, x, singleCoord[0], sampleWidth, singleCoord[1], metadata[int(results[0][0])-1]])

            sampleWidth = sampleWidth + 1
        # here I decide which is the best 'guess':
        if len(lettersWidth) > 0:
            bestGuess = sortGuesses(lettersWidth)
        else:
            break

        # string = metadata[int(bestGuess[5])-1]
        textValue.append(bestGuess[5])
        cv2.putText(out,bestGuess[5],(int(bestGuess[1]), int(bestGuess[2]+bestGuess[4])),0,1,(0,0,255))
        cv2.rectangle(out,(int(bestGuess[1]),int(bestGuess[2])),(int(x+bestGuess[3]), int(bestGuess[2]+bestGuess[4])),(0,0,0),1)
        # cv2.imshow('out',out)
        # cv2.waitKey(0)
        # reset start x value:
        x = x + bestGuess[3] - 2

        sampleWidth = 13
        # print(lettersWidth)	
        continue

	# # get the closest match of the word:
    word = "".join(textValue)
    print("word", word)
    something = difflib.get_close_matches(word, dictionary, n=3, cutoff= 0.5) 
    print(something)
    # cv2.imshow('im',eroded)
    cv2.imshow('out',out)
    cv2.waitKey(0)

def sortGuesses(letters):
    closest = letters[0]
    # sorted = np.sort(letters, axis = 0)
    # sorted = letters.sort()
    letters.sort(key=lambda x: np.sum(x[0]), reverse=False)
    # print(letters)
    for single in letters:
    	# print("sum of the closest is:", np.sum(single[0]))
    	# print(single[5])
    	if np.sum(single[0]) < np.sum(closest[0]):
    		closest = single
    # print("vienas rūšiavimas baigtas.")
    return closest  



# get the guess from the rate of interests:
def getRoi(eroded, coords):

    global model
    randomLetters = []
    x,y,w,h = [coords[0], coords[1], coords[2], coords[3]]
    roi = eroded[y:y+h, x:x+w]
    roismall = cv2.resize(roi,(10,30))
    # cv2.imshow("trial", roi)
    # cv2.imshow("roismall", roismall)
    # cv2.waitKey(0)
    roismall = roismall.reshape((1,300))
    roismall = np.float32(roismall)
    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
    # print("distances are; ", dists)
    # print("knn results are: ", results)
    # print("neighbor response: ", neigh_resp)

    # print("results are: ", results)
    for single in neigh_resp[0]:
    	random = []
    	# for singleLetters in single:
    		# jei daugiau nei 5, tada pridedam:
    		# random.append(chr(int(singleLetters)))
    	# print(metadata[int(single)-1])
    	# print(random)
    # print("neighour response is: ",neigh_resp)
    # print("ret value is: ", retval)
    return results, dists


def removeInnerContours(eroded, bounding, im_bw):
    # remove contours from inside of other contours:
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

    return newbounding

testing()