import cv2
import numpy as np
import sys

def testing():
    #######   training part    ############### 
    samples = np.loadtxt('generalsamples1.data',np.float32)
    responses = np.loadtxt('generalresponses1.data',np.float32)
    print(responses)
    responses = responses.reshape((responses.size,1))

    print(responses)
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    ############################# testing part  #########################

    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('sample_words.png')
    # img = cv2.resize(img, (640, 360))
    newx,newy = img.shape[1]/3,img.shape[0]/3     #new size (w,h)
    print("Rescaled, new dimensions: ", newx, newy)
    newimage = cv2.resize(img,(int(newx), int(newy)))
    out = newimage

    img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    eroded = cv2.erode(im_bw,kernel,iterations = 3)
    # cv2.imshow("Image", eroded)
    # cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []
    # classification
    samples =  np.empty((0,100))
    responses = []

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
        print([w,h])
        # let's try to adjust the width here:
        adjustWidth([x,y,w,h], eroded, maxHeight)
        cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)
        # roi = eroded[y:y+h,x:x+w]
        # roismall = cv2.resize(roi,(10,10))
        # roismall = roismall.reshape((1,100))
        # roismall = np.float32(roismall)
        # retval, results, neigh_resp, dists = model.findNearest(roismall, k = 2)
        # # print("retrieved value is: ", retval)
        # # print("results are: ", results)
        # # print("Neighbor responses are: ", neigh_resp)
        # # print("Distances from input vectors: ", dists)
        # string = str(int((results[0][0])))
        # cv2.putText(out,string,(x,y+h),0,1,(255,0,0))

    cv2.imshow('im',eroded)
    cv2.imshow('out',out)
    cv2.waitKey(0)

# extract the words into letters:
def adjustWidth(coords, eroded, maxHeight):
	
	[x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]

	sampleWidth = 25
	# positions: a j N J. Default - a
	heightPos = 1
	while sampleWidth < 65 :
		# here we should find the nearest neighbours, thickering the width +2px every iteration:
		cropToLetter = eroded[y:y+h, x:x+sampleWidth]
		cv2.imshow("cropped", cropToLetter)
		key = cv2.waitKey(0)

		if key == 27:
			sys.exit()
		elif key == 83:
			sampleWidth += 2
			continue
		# up arrow
		elif key == 82:
			y = y - (maxHeight - h)
			h = maxHeight
			continue
		# down arrow
		elif key == 84:
			y = y + (maxHeight - h)
			h = maxHeight
			continue
		# go to previous width:
		elif key == 81:
			sampleWidth -= 2
			continue
		else:
			submitKey = key
			print(chr(key))
			print(key)
			# reset start x value:
			x = x + sampleWidth
			sampleWidth = 25
			continue

            # responses.append(int(chr(key)))
            # sample = roismall.reshape((1,100))
            # samples = np.append(samples,sample,0)
		# im2, contours, hierarchy = cv2.findContours(cropToLetter,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		# for contour in contours:
		# 	[x,y,w,h] = cv2.boundingRect(contour)
		# 	cv2.rectangle(cropToLetter,(x,y),(x+w,y+h),(0,0,0),2)
		# 	# roi = cropToLetter[y:y+h,x:x+w]
		# 	# roismall = cv2.resize(roi,(10,10))
		# 	# roismall = roismall.reshape((1,100))
		# 	# roismall = np.float32(roismall)
		# 	# retval, results, neigh_resp, dists = model.findNearest(roismall, k = 2)
		# 	# print("retrieved value is: ", retval)
		# 	# print("results are: ", results)
		# 	# print("Neighbor responses are: ", neigh_resp)
		# 	# print("Distances from input vectors: ", dists)
		# 	# string = str(int((results[0][0])))
		# 	# cv2.putText(out,string,(x,y+h),0,1,(255,0,0))
		# cv2.imshow('im',cropToLetter)
  #   	# cv2.imshow('out',out)
		# cv2.waitKey(0)



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