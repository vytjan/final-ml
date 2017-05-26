import os
from PIL import Image
from array import *
from random import shuffle


data_image = array('B')
data_label = array('B')

FileList = [1]

Im = Image.open("175.png")

pixel = Im.load()
# print(pixel)

width, height = Im.size
# print(width, height)

for x in range(0,height):

	for y in range(0,width):
		# print(pixel[y,x])
		data_image.append(pixel[y,x])

# data_label.append(label) # labels start (one unsigned byte each)

hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

# header for label array

header = array('B')
header.extend([0,0,8,1,0,0])
header.append(int('0x'+hexval[2:][:2],16))
header.append(int('0x'+hexval[2:][2:],16))
print(header)

data_label = header + data_label

# additional header for images array

if max([width,height]) <= 256:
	header.extend([0,0,0,width,0,0,0,height])
else:
	raise ValueError('Image exceeds maximum size: 256x256 pixels');

header[3] = 3 # Changing MSB for image data (0x00000803)

data_image = header + data_image
print(data_image[1])
# print(header)

# output_file = open('test2', 'wb')
# data_image.tofile(output_file)
# output_file.close()

# output_file = open(name[1]+'text-labels-ubyte', 'wb')
# data_label.tofile(output_file)
# output_file.close()

# gzip resulting files

# for name in Names:
os.system('gzip '+'test')
# 	os.system('gzip '+name[1]+'-labels-idx1-ubyte')