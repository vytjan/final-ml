#import modules
import sys
import tensorflow as tf
from PIL import Image,ImageFilter

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 300])
    W = tf.Variable(tf.zeros([300, 61]))
    b = tf.Variable(tf.zeros([61]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    """
    Load the model.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.
    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        print ("Model restored.")
        from matplotlib import pyplot as plt
        # prediction=tf.argmax(y,1)
        # return prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        classification = sess.run(tf.argmax(y, 1), feed_dict={x: [imvalue]})
        print(imvalue)
        # plt.imshow(imvalue.reshape(30, 10), cmap=plt.cm.binary)
        # plt.show()
        print('NN predicted', classification[0])


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    print(argv)
    newImage = Image.open(argv).convert('L')
    # width = float(im.size[0])
    # height = float(im.size[1])
    # newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    # if width > height: #check which dimension is bigger
    #     #Width is bigger. Width becomes 20 pixels.
    #     nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
    #     if (nheigth == 0): #rare case but minimum is 1 pixel
    #         nheigth = 1  
    #     # resize and sharpen
    #     img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
    #     newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    # else:
    #     #Height is bigger. Heigth becomes 20 pixels. 
    #     nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
    #     if (nwidth == 0): #rare case but minimum is 1 pixel
    #         nwidth = 1
    #      # resize and sharpen
    #     img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
    #     newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    print(tv)
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    # print(tva)  
    return tva
    #print(tva)

def main(argv):
    """
    Main function.
    """
    imvalue = imageprepare(argv)
    # print(imvalue)
    predint = predictint(imvalue)
    print (predint) #first value in list
    
if __name__ == "__main__":
    main(sys.argv[1])