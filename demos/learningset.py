import cv2
import numpy as np
import sys
import os

# classification sets
# global samples
# global filename
# filename = 0
samples =  np.empty((0,300))
# global responses
responses = []


# iterate over the the folders of letters:
directory = 'letters-50'
for foldername in os.listdir(directory):
    # if filename.endswith("") or filename.endswith(".py"): 
    # print(os.path.join(directory, filename))
    print(foldername)
    classType = foldername
    # iterate over 10x30 png images in every folder"
    for filename in os.listdir(os.path.join(directory, foldername)):
        print(filename)
        roismall = cv2.imread(directory + '/' + foldername + '/' + filename)
        img = cv2.cvtColor(roismall,cv2.COLOR_BGR2GRAY)
        responses.append(int(classType))
        sample = img.reshape((1,300))
        # print(sample)
        # print("Int is: ", int(classType))
        # print("image is: ", sample)
        samples = np.append(samples,sample,0)
        # cv2.imshow("thersh", img)
        # cv2.imshow("model", roismall)
        # cv2.waitKey(0)
print(responses)
responses = np.array(responses,np.float32)
print(responses)
responses = responses.reshape((responses.size,1))

np.savetxt('generalsamplesNEW.data',samples)
np.savetxt('generalresponsesNEW.data',responses)
print ("training complete")