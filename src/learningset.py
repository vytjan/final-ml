import cv2
import numpy as np
import sys
import os

samples =  np.empty((0,300))
responses = []

# iterate over the the folders containing letters:
directory = 'letters-50'
for foldername in os.listdir(directory):
    print(foldername)
    classType = foldername
    # iterate over 10x30 png images in every folder:
    for filename in os.listdir(os.path.join(directory, foldername)):
        print(filename)
        roismall = cv2.imread(directory + '/' + foldername + '/' + filename)
        img = cv2.cvtColor(roismall,cv2.COLOR_BGR2GRAY)
        # a letter tag is the name of a folder:
        responses.append(int(classType))
        sample = img.reshape((1,300))
        samples = np.append(samples,sample,0)
print(responses)
responses = np.array(responses,np.float32)
print(responses)
responses = responses.reshape((responses.size,1))

# save learning data set to the new files:
np.savetxt('generalsamples2.data',samples)
np.savetxt('generalresponses2.data',responses)
print ("training complete")