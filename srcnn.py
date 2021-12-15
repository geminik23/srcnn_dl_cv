import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
import skimage.metrics

## !!!!!!!!!!!!
## used scipy==1.1.0 to use the imread
## tensorflow==1.15
def imread(path, is_grayscale=True):
    """
    read image using path
    default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
    """
    scale down and up
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def preprocess(path, scale=3):
    """
    preprocess image file 

    RETURN (input_, label_)
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """
    # read image from path
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # normalize
    image = image / 255.
    label_ = label_ / 255.

    # apply image file with bicubic interpolation
    input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

    return input_, label_


### image hyper parameters
c_dim = 1
input_size = 255

### the model weights and biases 
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }


# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3']

### Load the pre-trained model file
model_path='./model/model.npy'

model = np.load(model_path, encoding='latin1', allow_pickle=True).item()

def visualize_the_weight(key, idx_channel = 0):
    # reshape to 3D -> height, width and the number of channels(? multiplication input with output channels)
    w_shape = weights[key].get_shape()
    reshaped_model = model[key].reshape(w_shape[0], w_shape[1], w_shape[2]*w_shape[3])

    # cannot visualize all weights because of too many channels. so this function shows only one channel
    plt.imshow(reshaped_model[:,:,idx_channel])
    plt.show()


visualize_the_weight('w1', 0)
visualize_the_weight('w2', 0)
visualize_the_weight('w3', 0)
  



### initialize the model variabiles with the pre-trained model file 
sess = tf.Session()
for key in weights.keys(): sess.run(weights[key].assign(model[key]))
for key in biases.keys(): sess.run(biases[key].assign(model[key]))



####
##################TEST
# read the image
blurred_image, groudtruth_image = preprocess('./image/butterfly_GT.bmp')

# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)
output_ = sess.run(conv3, feed_dict={inputs: input_})

scipy.misc.imsave('image/blurred.bmp', blurred_image)
scipy.misc.imsave('image/groundtruth.bmp', groudtruth_image)

out = output_.reshape(output_.shape[1], output_.shape[2])
# (1, 243, 243, 1)
scipy.misc.imsave('image/output.bmp', out)

### compute the psnr
# Due to different image size, I cropped the blurred image same with output size.
print("PSNR between blurred and SR images is {}".format(skimage.metrics.peak_signal_noise_ratio(out, blurred_image[6:255-6, 6:255-6])))
