import cv2
import numpy as np
import sys
import os
import time


# classification sets
global samples
# global filename
# filename = 0
samples =  np.empty((0,300))
global responses
responses = []

def testing():
	global responses
	global samples
	kernel = np.ones((2,2),np.uint8)
	img = cv2.imread(sys.argv[1])
	newx,newy = img.shape[1],img.shape[0]     #new size (w,h)
	print("Rescaled, new dimensions: ", newx, newy)
	newimage = cv2.resize(img,(int(newx), int(newy)))
	out = newimage

	img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)

	thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,15,24)

	cv2.imshow("threshold", thresh1)
	cv2.waitKey(0)

	kernel2 = np.ones((2,2),np.uint8)		

	eroded = cv2.erode(thresh1,kernel2,iterations = 2)
	# eroded = cv2.dilate(eroded,kernel,iterations = 1)
	cv2.imshow("eroded", eroded)
	cv2.waitKey(0)
	im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# list of bounding rects
	bounding = []

	for cnt in contours:
	    [x,y,w,h] = cv2.boundingRect(cnt)
	    if (w <20 and h<20) or (h > 180):
	        continue
	    bounding.append([x,y,w,h])
	    #if we need to check which contours are detected:
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
	    #adjust the width here:
	    adjustWidth([x,y,w,h], eroded, maxHeight)


	# responses = np.array(responses,np.float32)
	# responses = responses.reshape((responses.size,1))
	# print ("training complete")

 #    # Save learning data to text files:
	# np.savetxt('generalsamplesA.data',samples)
	# np.savetxt('generalresponsesA.data',responses)
	# cv2.imshow('im',eroded)
	# cv2.imshow('out',out)
	# cv2.waitKey(0)

# extract the words into letters:
def adjustWidth(coords, eroded, maxHeight):
	# global filename
	global samples
	[x,y,w,h] = [coords[0], coords[1], coords[2], coords[3]]
	initx = x
	inity = y
	inith = h
	contourWidth = x + w
	sampleWidth = 16
	# positions: a j N J. Default - a
	heightPos = 1
	while sampleWidth < 60 and x+sampleWidth < contourWidth + 5 and x >= 0 and y >=0:
		# adjust the image, thickering the width +2px every iteration:
		
		cropToLetter = eroded[y:y+h, x:x+sampleWidth]

		cv2.imshow("cropped", cropToLetter)
		key = cv2.waitKey(0)

		# * 42; - 45
		if key == 190:
			# reset the letter:
			print("before delete: ", len(samples))
			samples[:-1]
			print("after delete: ", len(samples))
			continue
		elif key == 91:
			h -= 1
			y-=1
			continue
		elif key == 93:
			h+=1
			continue
		elif key == 39:
			h += 1
			y+=1
			continue
		elif key == 92:
			h-=1
			continue
		elif key == 27:
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
			if x - 2 > 0:
				x -= 2
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
			sampleWidth = 16
			continue
		else:
			if key == 226:
				key = chr(cv2.waitKey(0))
				key = ord(key.upper())
			else:
				print(key)
				
			print(chr(key))
			roi = eroded[y:y+h,x:x+sampleWidth]
			
			roismall = cv2.resize(roi,(10,30))

			filename = str(round(time.time())) 	
			print(os.path.join('letters-50', str(key) + filename + '.png'))
			if not os.path.exists(os.path.join('letters', str(key))) :
				print("create a new directory: ", str(key))
				os.mkdir(os.path.join('letters-50', str(key)))
			print(os.path.join('letters', str(key), filename + '.png'))
			cv2.imwrite(os.path.join('letters-50', str(key), filename + '.png'), roismall)
		
			cv2.imshow("model", roismall)
			cv2.waitKey(0)
			responses.append(int(key))
			# print("shape is: ", roismall.shape)
			sample = roismall.reshape((1,300))
			samples = np.append(samples,sample,0)
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