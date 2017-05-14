import cv2
import numpy as np
import sys
from select import select


# classification sets
global samples
samples =  np.empty((0,300))
global responses
responses = []

def testing():
	global responses
	global samples
	kernel = np.ones((2,2),np.uint8)
	img = cv2.imread('test4.png')
	newx,newy = img.shape[1]/3,img.shape[0]/3     #new size (w,h)
	print("Rescaled, new dimensions: ", newx, newy)
	newimage = cv2.resize(img,(int(newx), int(newy)))
	out = newimage

	img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
	# img = cv2.medianBlur(img,5)

	# (thresh, thresh1) = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
	# cv2.imshow("medianblur", im_bw)
	# cv2.waitKey(0) 

	thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,9,16)

	# blur = cv2.GaussianBlur(img,(4,4),0)
	# ret3,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
	cv2.imshow("threshold", thresh1)
	cv2.waitKey(0)

	kernel2 = np.ones((2,2),np.uint8)
	# dilation = cv2.dilate(thresh1,kernel,iterations = 1)
	# cv2.imshow("dilated", dilation)
	# cv2.waitKey(0)
	eroded = cv2.erode(thresh1,kernel2,iterations = 2)
	# eroded = cv2.dilate(eroded,kernel,iterations = 1)
	cv2.imshow("eroded", eroded)
	cv2.waitKey(0)
	im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# list of bounding rects
	bounding = []

	# remove the outer contour:
	# contours.pop(0)


	for cnt in contours:
	    [x,y,w,h] = cv2.boundingRect(cnt)
	    if (w < 20 and h<20) or (h > 180):
	        continue
	    bounding.append([x,y,w,h])
	    # cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)

	cv2.imshow("contours", eroded)
	cv2.waitKey(0)

	newbounding = removeInnerContours(eroded, bounding, thresh1)
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
	    # cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)
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

	responses = np.array(responses,np.float32)
	responses = responses.reshape((responses.size,1))
	print ("training complete")

	np.savetxt('generalsamplesA.data',samples)
	np.savetxt('generalresponsesA.data',responses)
	cv2.imshow('im',eroded)
	cv2.imshow('out',out)
	cv2.waitKey(0)

# extract the words into letters:
def adjustWidth(coords, eroded, maxHeight):
	global samples
	[x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]

	contourWidth = x + w
	sampleWidth = 16
	# positions: a j N J. Default - a
	heightPos = 1
	while sampleWidth < 45 and x+sampleWidth < contourWidth + 5 and x >= 0 and y >=0:
		# here we should find the nearest neighbours, thickering the width +2px every iteration:
				# here I need to put the contour of the letter to the center of the 30x10:
		
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
			y = y - 0.5*h
			if(y < 0):
				y = 0
			h = 1.5*h
			continue
		# down arrow
		elif key == 84:
			# y = y + (maxHeight - h)
			h = 1.5*h
			continue
		# go to previous width:
		elif key == 81:
			sampleWidth -= 2
			continue
		# if a letter:
		elif key == 32:
			y = y - h
			if(y < 0):
				y = 0
			h = 3*h
			continue
		elif key == 13:
			x = x+sampleWidth
			sampleWidth = 20
			continue
		else:
			if key == 226:
				key = chr(cv2.waitKey(0))
				key = ord(key.upper())
			else:
				# tempImage = eroded[y:y+h, x:x+sampleWidth]
				print(key)
		# cntrs.pop(0)
		# for cnt in cntrs:
		# 	[a,b,c,d] = cv2.boundingRect(cnt)
  #   		# if (w < 20 and h<20) or (h > 180):
  #       		# continue
		# 	print("bounding rectangle: ",[a,b,c,d])
  #   		# bounding.append([x,y,w,h])
			# rect = cv2.minAreaRect(cntrs[0])
			# box = cv2.boxPoints(rect)
			# box = np.int0(box)
			# print("box is: ", box)
			# cv2.drawContours(cropToLetter,[box],0,(0,0,255),1)
			# cv2.rectangle(tempImage,(a,b),(a+c,b+d),(0,0,0),3)

		# 	cv2.imshow("contours partly", newIm)
		# 	cv2.waitKey(0)
		# 	# print(x+sampleWidth)
			# [a,b,c,d] = cv2.boundingRect(cntrs[0])
			# cv2.rectangle(cropToLetter,(a,b),(a+c,b+d),(0,0,0),3)
			# s = input("Write a capital letter ")
			# print(s)
			# submitKey = key
			# print(chr(key))
				
			print(chr(key))	
			# roi = eroded[y:b+d,x:a+c]
			bordersize=1
			border=cv2.copyMakeBorder(cropToLetter, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
			newIm, cntrs, hierarchy = cv2.findContours(border,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cntrs.pop(0)
			c = max(cntrs, key = cv2.contourArea)
			print("length of cnts",len(cntrs))
			[a,b,c,d] = cv2.boundingRect(c)
			cv2.rectangle(border,(a,b),(a+c,b+d),(0,0,0),3)
			cv2.imshow("contours", border)
			cv2.waitKey(0)
			# roi = eroded[y:y+h,x:x+sampleWidth]


			roismall = cv2.resize(roi,(10,30))
			cv2.imshow("model", roismall)
			cv2.waitKey(0)
			responses.append(int(key))
			sample = roismall.reshape((1,300))
			samples = np.append(samples,sample,0)
			# cv2.imshow('Added to classification',cropToLetter)
			# cv2.waitKey(0)
			# reset start x value:
			x = x + sampleWidth
			sampleWidth = 16
			continue


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