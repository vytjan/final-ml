import numpy 
import gzip
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

W = tf.Variable(tf.zeros([300,62]))
b = tf.Variable(tf.zeros([62]))

x = tf.placeholder(tf.float32, shape=[None, 300])
y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32, shape=[None, 62])



sess.run(tf.global_variables_initializer())


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for _ in range(3050):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'my_test_model', global_step=1000)
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# print(sess.run(y, feed_dict={x: mnist.test.images	}))

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
# print(mnist.test.images.shape)
img = mnist.test.images[num]
print(img)


feed_dict = {x: [img]}
classification = sess.run(y, feed_dict)
print(classification.argmax(axis=1))
print(classification)
print(len(classification[0]))
print(numpy.amax(classification, axis=1))	
# temp = numpy.argpartition(-classification, classification.argmax(axis=1))
# result_args = -temp

# temp = numpy.partition(-classification, classification.argmax(axis=1))
# result = -temp
# print(result_args)
# print(result)
# idx = numpy.argpartition(classification,range(4))[:,:-4-1:-1] # topM_ind
# out = classification[numpy.arange(classification.shape[0])[:,None],idx]    # topM_score


# print(y)
# ind_max = tf.argmax(x, 1)
# print(ind_max)
# flat_y = tf.reshape(y, [-1])  # Reshape to a vector.
# print(flat_y)
# N.B. Handles 2-D case only.
# flat_ind_max = ind_max + tf.cast(tf.range(tf.shape(y)[0]) * tf.shape(y)[1], tf.float32)

# y_ = tf.gather(flat_y, flat_ind_max)

# print(sess.run(y_)) # ==> "array([2, 6], dtype=int32)"
# print(ind_max)
# x_max = tf.reduce_max(y, reduction_indices=[1])
# print(sess.run(x_max))
# print(y.eval(feed_dict))
# classification = sess.run(tf.argmax(y,1), feed_dict={x: [img]})
# print(tf.maximum(y,1))
# y_max = tf.reduce_max(y, reduction_indices=[1])
# print(sess.run(y_max))
# classification = sess.run(tf.global_variables(), feed_dict={x: [img]})
# print(accuracy.eval(y, feed_dict={x: img}))
# probabilities=1
# print("printrobabilities ", probabilities.eval(y, feed_dict={x: img}, session=sess))
plt.imshow(img.reshape(30, 10), cmap=plt.cm.binary)
plt.show()
# print('NN predicted', classification)


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

	# with open("test.gz", 'rb') as f:
	# 	test_image = extract_images(f)	
	# 	images
	# 	# Convert shape from [num examples, rows, columns, depth]
	#       # to [num examples, rows*columns] (assuming depth == 1)
	# 	assert images.shape[3] == 1
	# 	images = images.reshape(images.shape[0],
	# 	images.shape[1] * images.shape[2])
	# 	if dtype == dtypes.float32:
	# 	# Convert from [0, 255] -> [0.0, 1.0].
	# 	images = images.astype(numpy.float32)
	# 	images = numpy.multiply(images, 1.0 / 255.0)
	# 	print(images)

	# 	classification = sess.run(tf.argmax(y, 1), feed_dict={x: [test_image]})
	# 	plt.imshow(img.reshape(30, 10), cmap=plt.cm.binary)
	# 	plt.show()
	# 	print('NN predicted', classification[0])
