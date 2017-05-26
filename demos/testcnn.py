# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST expert tutorial by Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do
This script is based on the Tensoflow MNIST expert tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
"""

#import modules
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

#import data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# sess = tf.InteractiveSession()

#import modules
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

from matplotlib import pyplot as plt
from random import randint

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
    print(classification.argmax(axis=1))
    reply = []
    reply.append(classification.argmax(axis=1))
    reply.append(np.amax(classification, axis=1)) 
    # print(classification)
    # print(len(classification[0]))
    print(np.amax(classification, axis=1))
    # print(classification[0])
    return reply


def imageprepare(argv):

  im = Image.open(argv).convert('L')

  tv = list(im.getdata()) #get pixel values
  # print(image)
  # data = numpy.empty((0,300))
  # sample = tv.reshape((1,300))
   # = np.append(samples,sample,0)
  # data = numpy.frombuffer(buf, dtype=numpy.uint8)
  #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  # print(tv)
  tva = [ (255-x)*1.0/255.0 for x in tv]
  data = np.array(tva)
  print(data)
  newData = 1 - data
  proceeded = newData
  # newData = np.invert(data)
  # print(data)
  plt.imshow(proceeded.reshape(30, 10), cmap=plt.cm.binary)
  plt.show() 
  # print(tva)
  return proceeded


def main(argv):
  """
  Main function.
  """
  imvalue = imageprepare(argv)
  predint = predictint(imvalue)
  return predint
  # print(predint)
  # print (predint[0]) #first value in list
    
if __name__ == "__main__":
  main(sys.argv[1])