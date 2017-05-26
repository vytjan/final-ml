#import modules
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

from matplotlib import pyplot as plt
from random import randint
import cv2
import numpy as np
import difflib
import codecs

#######   training part    ###############
global samples
global responses
global dictionary
global metadata

# load the dictionary:
dictionary = codecs.open("dictionary.dat", encoding="utf-8").read().splitlines()
metadata = codecs.open("batches.meta.txt", encoding="utf-8").read().splitlines()
print(metadata)
# print(len(dictionary))

# something = difflib.get_close_matches('o6ia', dictionary)
# print("zodis yra ",something)

samples = np.loadtxt('generalsamplesA.data',np.float32)
responses = np.loadtxt('generalresponsesA.data',np.float32)
# print(responses)
responses = responses.reshape((responses.size,1))

# print(responses)
global model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

def testing():


  # # Create the model
  # x = tf.placeholder(tf.float32, [None, 300])
  # y_ = tf.placeholder(tf.float32, [None, 62])
  # W = tf.Variable(tf.zeros([300, 62]))
  # b = tf.Variable(tf.zeros([62]))
  # y = tf.nn.softmax(tf.matmul(x, W) + b)

  # def weight_variable(shape):
  #   initial = tf.truncated_normal(shape, stddev=0.1)
  #   return tf.Variable(initial)

  # def bias_variable(shape):
  #   initial = tf.constant(0.1, shape=shape)
  #   return tf.Variable(initial)


  # def conv2d(x, W):
  #   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # def max_pool_2x2(x):
  #   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
  #                         strides=[1, 2, 2, 1], padding='SAME')


  # W_conv1 = weight_variable([5, 5, 1, 32])
  # b_conv1 = bias_variable([32])

  # x_image = tf.reshape(x, [-1,10,30,1])
  # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # h_pool1 = max_pool_2x2(h_conv1)


  # W_conv2 = weight_variable([5, 5, 32, 64])
  # b_conv2 = bias_variable([64])

  # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  # h_pool2 = h_conv2

  # W_fc1 = weight_variable([5 * 15 * 32, 1024])
  # b_fc1 = bias_variable([1024])

  # h_pool2_flat = tf.reshape(h_pool1, [-1, 5*15*32])
  # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # keep_prob = tf.placeholder(tf.float32)
  # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # W_fc2 = weight_variable([1024, 62])
  # b_fc2 = bias_variable([62])

  # y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


  # init_op = tf.global_variables_initializer()
  # saver = tf.train.Saver()

    ############################# testing part  #########################

    kernel = np.ones((2,2),np.uint8)

    img = cv2.imread('skrenda.png')
    # img = cv2.resize(img, (640, 360))
    newx,newy = img.shape[1]/3.5,img.shape[0]/3.5    #new size (w,h)
    # print("Rescaled, new dimensions: ", newx, newy)
    newimage = cv2.resize(img,(int(newx), int(newy)))
    # out = newimage
    # cv2.imshow("color image", out)
    # cv2.waitKey(0)

    img = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY) 
    # cv2.imshow("grayscale", img)
    # cv2.waitKey(0)
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
      cv2.THRESH_BINARY,21,21)


    kernel2 = np.ones((2,2),np.uint8)
    eroded = cv2.erode(thresh1,kernel2,iterations = 1)
    im2, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # list of bounding rects
    bounding = []

  # contours.pop(0)  

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (w < 20 and h<20) or (h > 180):
            continue
        bounding.append([x,y,w,h])
        # cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,0),2)

    # cv2.imshow("contours", eroded)
    # cv2.waitKey(0)

    newbounding = removeInnerContours(eroded, bounding, newImage)

    maxHeight = newbounding[0][3]
    for a in newbounding:
        if a[3] > maxHeight:
            maxHeight = a[3]
    print("max height is: ", maxHeight)

    for single in newbounding:
        [x,y,w,h] = [single[0], single[1], single[2], single[3]]
        # print([x,y,w,h])
        adjustWidth([x,y,w,h], eroded, maxHeight, out, newx, newy)
        # with tf.Session() as sess:
        #   sess.run(init_op)
        #   saver.restore(sess, "./model2.ckpt")
    #print ("Model restored.")
    
    # prediction=tf.argmax(y_conv,1)
    # return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)

    print("finished. ")

    # # prediction=tf.argmax(y_conv,1)
    
    # feed_dict = {x: [imvalue],keep_prob: 1.0}
    # classification = sess.run(y_conv, feed_dict)
    # # classification = prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
    # print(classification.argmax(axis=1)[0])
    # reply = []
    # reply.append(classification.argmax(axis=1)[0])
    # reply.append(np.amax(classification, axis=1)[0]) 
    # # print(classification)
    # # print(len(classification[0]))
    # print(np.amax(classification, axis=1)[0])
    # # print(classification[0])
    # return reply
        # cv2.rectangle(eroded,(x,y),(x+w,y+h),(0,0,0),2)

    # cv2.imshow('im',eroded)
    # cv2.imshow('out',out)
    # cv2.waitKey(0)

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
        while sampleWidth < 45 and x + sampleWidth < contourWidth+5:

            # get results of the height variations of the letters:
            heightVariations = []

            # clean up this stuff later:------------------------------------------------------------------
            # If J, do nothing:
            heightVariations.append([y,h])
            # if l, increase below:
            # if h*1.5 < 85:
            height = h
            # while height < 70:
                # print(height)
            heightVariations.append([y,height])
                # height += 5
                # if y-h/2 > 0 and h < 70:
                #     heightVariations.append([y-height/2, height])
                # elif y-h > 0 and h < 70:
                #     heightVariations.append([y-height, height])
                # height += 10


            # print("length of heightvars: ", len(heightVariations))
            for singleCoord in heightVariations:
                # print("new iteration of getting roi: \n")
                # xplus = 0
                # while xplus < 7:
                results, dists = getRoi(eroded, [x, singleCoord[0], sampleWidth, singleCoord[1]])
                lettersWidth.append([dists, x, singleCoord[0], sampleWidth, singleCoord[1], results])

            sampleWidth = sampleWidth + 2

        # here I decide which is the best 'guess':
        # print("length is: ", len(lettersWidth))
        if len(lettersWidth) > 0:
            bestGuess = sortGuesses(lettersWidth)
        else:
            break

        print("best guess looks like: ", bestGuess[5])
        string = metadata[bestGuess[5]-1]
        textValue.append(string)
        cv2.putText(out,string,(int(bestGuess[1]), int(bestGuess[2]+bestGuess[4])),0,1,(0,0,255))
        cv2.rectangle(out,(int(bestGuess[1]),int(bestGuess[2])),(int(x+bestGuess[3]), int(bestGuess[2]+bestGuess[4])),(0,0,0),1)
        # reset start x value:
        x = x + bestGuess[3]

        sampleWidth = 13
        # print(lettersWidth)
        continue

  # # get the closest match of the word:
    word = "".join(textValue)
    print("word", word)
    something = difflib.get_close_matches(word, dictionary, n=2, cutoff= 0.5) 
    print(something)
    cv2.imshow('im',eroded)
    cv2.imshow('out',out)
    cv2.waitKey(0)

def sortGuesses(letters):
    closest = letters[0]
    for single in letters:
      # single[5] = metadata[single[5] - 1]
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
    randomLetters = []
    x,y,w,h = [coords[0], coords[1], coords[2], coords[3]]
    roi = eroded[y:y+h, x:x+w]
    roismall = cv2.resize(roi,(10,30))
    # cv2.imshow("trial", roi)
    # cv2.imshow("roismall", roismall)
    # cv2.waitKey(0)
    roismall = roismall.reshape((1,300))
    # print(roismall[0])
    roismall = np.float32(roismall)
    # retval, results, neigh_resp, dists = model.findNearest(roismall, k = 3)
    # print("distances are; ", dists)
    # print("knn results are: ", chr(int(results[0][0])))

    readyimage = imageprepare(roismall[0])
    reply = predictint(readyimage)
    print(metadata[reply[0]-1])
    results = reply[0]
    dists = reply[1]
    # print("results are: ", results)
    # for single in neigh_resp:
      # random = []
      # for singleLetters in single:
        # jei daugiau nei 5, tada pridedam:
        # random.append(chr(int(singleLetters)))
        # print(chr(int(singleLetters)))
      # print(random)
    # print("neighour response is: ",neigh_resp)
    # print("ret value is: ", retval)
    return results, dists

def predictint(imvalue):

  # Create the model
  x = tf.placeholder(tf.float32, [None, 300])
  y_ = tf.placeholder(tf.float32, [None, 62])
  W = tf.Variable(tf.zeros([300, 62]))
  b = tf.Variable(tf.zeros([62]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,10,30,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)


  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = h_conv2

  W_fc1 = weight_variable([5 * 15 * 32, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool1, [-1, 5*15*32])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 62])
  b_fc2 = bias_variable([62])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()  
      
  """
  Load the model2.ckpt file
  file is stored in the same directory as this python script is started
  Use the model to predict the integer. Integer is returend as list.
  Based on the documentatoin at
  https://www.tensorflow.org/versions/master/how_tos/variables/index.html
  """

  with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "./model2.ckpt")
    #print ("Model restored.")
    
    # prediction=tf.argmax(y_conv,1)
    # return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)



    # prediction=tf.argmax(y_conv,1)
    
    feed_dict = {x: [imvalue],keep_prob: 1.0}
    classification = sess.run(y_conv, feed_dict)
    # classification = prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
    # print(classification.argmax(axis=1)[0])
    reply = []
    reply.append(classification.argmax(axis=1)[0])
    reply.append(np.amax(classification, axis=1)[0]) 
    # print(classification)
    # print(len(classification[0]))
    # print(np.amax(classification, axis=1)[0])
    # print(classification[0])
  tf.reset_default_graph()
  return reply


def imageprepare(img):

  # im = Image.open(argv).convert('L')

  # tv = list(im.getdata()) #get pixel values
  # print(image)
  # data = numpy.empty((0,300))
  # sample = tv.reshape((1,300))
   # = np.append(samples,sample,0)
  # data = numpy.frombuffer(buf, dtype=numpy.uint8)
  #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  # print(tv)
  tva = [ (255-x)*1.0/255.0 for x in img]
  data = np.array(tva)
  # print(data)
  newData = 1 - data
  proceeded = newData
  # newData = np.invert(data)
  # print(data)
  # plt.imshow(proceeded.reshape(30, 10), cmap=plt.cm.binary)
  # plt.show() 
  # print(tva)
  return proceeded


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


# def main(argv):
#   """
#   Main function.
#   """
#   imvalue = imageprepare(argv)
#   predint = predictint(imvalue)
#   return predint
#   # print(predint)
#   # print (predint[0]) #first value in list
    
# if __name__ == "__main__":
#   main(sys.argv[1])