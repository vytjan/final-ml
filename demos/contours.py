import numpy as np
import cv2

def sort_contours(cnts, method="left-to-right"):
  # initialize the reverse flag and sort index
  reverse = False
  i = 0

  # handle if we need to sort in reverse
  if method == "right-to-left" or method == "bottom-to-top":
    reverse = True

  # handle if we are sorting against the y-coordinate rather than
  # the x-coordinate of the bounding box
  if method == "top-to-bottom" or method == "bottom-to-top":
    i = 1

  # construct the list of bounding boxes and sort them from top to
  # bottom
  boundingBoxes = [cv2.boundingRect(c) for c in cnts]
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))

  # return the list of sorted contours and bounding boxes
  return (cnts, boundingBoxes)

def draw_contour(image, c, i):
  # compute the center of the contour area and draw a circle
  # representing the center
  M = cv2.moments(c)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
 
  # draw the countour number on the image
  cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (255, 255, 255), 2)
 
  # return the image with the contour number drawn on it
  return image

image = cv2.imread("sample_text.png")
accumEdged = np.zeros(image.shape[:2], dtype="uint8")
 
# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
  # blur the channel, extract edges from it, and accumulate the set
  # of edges for the image
  chan = cv2.medianBlur(chan, 11)
  edged = cv2.Canny(chan, 50, 200)
  accumEdged = cv2.bitwise_or(accumEdged, edged)
 
# show the accumulated edge map
cv2.imshow("Edge Map", accumEdged)

# find contours in the accumulated image, keeping only the largest
# ones
(im2, cnts, _) = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
  cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
orig = image.copy()
 
# loop over the (unsorted) contours and draw them
for (i, c) in enumerate(cnts):
  orig = draw_contour(orig, c, i)
 
# show the original, unsorted contour image
cv2.imshow("Unsorted", orig)
 
# sort the contours according to the provided method
(cnts, boundingBoxes) = sort_contours(cnts, "left-to-right")
 
# loop over the (now sorted) contours and draw them
for (i, c) in enumerate(cnts):
  draw_contour(image, c, i)
 
# show the output image
cv2.imshow("Sorted", image)
cv2.waitKey(0)
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)


# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(hierarchy)




#for index, item in enumerate(hierarchy[0]):
#        print(index, item)

#for cnt in contours:
    
 #   area = cv2.contourArea(cnt)
# remove inside contours and join nearby contours
    # if area < 1200 or area > 15000:
      #  continue
        
  #  print(area)
   # x,y,w,h = cv2.boundingRect(cnt)
   # cv2.rectangle(imgray,(x,y),(x+w,y+h),(0,255,0),2)

# cv2.drawContours(im, contours, -1, (0,255,0), 3)

#cv2.imshow('image' , imgray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
