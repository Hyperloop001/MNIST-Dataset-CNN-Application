### Provide functional supports to mnist algorithms ###
### Author: Bill Tong 
### Date: 2018-06-07

import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from PIL import Image

#------------------------------Convolutional Neural Network Functions------------------------------#
# weight_variable(shape, algoType):
#    Type: tf.shape, string ==> tf.Tensor
#    Input: shape of desired variable, algorithm type
#    Output: variable tensor
#    Side effects: none
#    Purposes: produce a corresponding weight variable
#    Note: algoType is either 'CNN' or 'LINEAR'
def weight_variable(shape, algoType):
    if algoType == 'CNN':
        initial = tf.truncated_normal(shape=shape, stddev = 0.1)   
        return tf.Variable(initial)  
    if algoType == 'LINEAR':
        initial = tf.zeros(shape=shape)
        return tf.Variable(initial)  
    else:
        printf("ERROR: no algorithm type was found")
        return None

# bias_variable(shape, algoType):
#    Type: tf.shape, string ==> tf.Tensor
#    Input: shape of desired variable, algorithm type
#    Output: variable tensor
#    Side effects: none
#    Purposes: produce a corresponding bias variable
#    Note: algoType is either 'CNN' or 'LINEAR'
def bias_variable(shape, algoType):
    if algoType == 'CNN':
        initial = tf.constant(0.1, shape=shape) 
        return tf.Variable(initial)  
    if algoType == 'LINEAR':
        initial = tf.zeros(shape=shape)
        return tf.Variable(initial)  
    else:
        printf("ERROR: no algorithm type was found")
        return None    

# conv2d(x, W):
#    Type: tf.Tensor, tf.Tensor ==> tf.Tensor
#    Input: data tensor and data filter
#    Output: filtered out data tensor
#    Side effects: none
#    Purposes: produce a data tensor from convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# max_poop_2x2(x):
#    Type: tf.Tensor ==> tf.Tensor
#    Input: data tensor 
#    Output: max pooled data tensor
#    Side effects: none
#    Purposes: produce a data tensor from max pooling layer
def max_poop_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#------------------------------Convolutional Neural Network Functions------------------------------#



#------------------------------Image Loading and Processing Functions------------------------------#
# readImg(path):
#    Type: string ==> tf.Tensor
#    Input: path to image file
#    Output: original image tensor
#    Side effects: none
#    Purposes: load a image from specified path as a data tensor, produce original image data tensor
def  readImg(path):
    imageRawData = tf.gfile.FastGFile(path,'rb').read()
    imgData = tf.image.decode_jpeg(imageRawData)      
    return imgData    
    
# normalizedReadImg(path, size=[28,28], invert=0):
#    Type: string, tf.size, int ==> tf.Tensor
#    Input: path to image file, size of desired output, invert color option
#    Output: grey scale image tensor with size and invert color
#    Side effects: none
#    Purposes: load a image from specified path as a data tensor, then greyscale it, convert type, and resize, produce manipulated image data tensor
def  normalizedReadImg(path, size=[28,28], invert=0):
    imgData = readImg(path)
    imgData = tf.image.rgb_to_grayscale(imgData)
    imgData = tf.image.convert_image_dtype(imgData, tf.float32) # Note: converting dtype happens before resizing for secure data
    imgData = tf.image.resize_images(imgData, size, method=0)     
    imgData = imgData[:,:,0]
    if invert==1:
        imgData = 1-imgData
    return imgData 

#------------------------------Image Loading and Processing Functions------------------------------#



#------------------------------Image Plot Scheme Functions------------------------------#
# transChannel(imgDataEval,to_color):
#    Type: npArray, int ==> npArray, plt.cm-method
#    Input: evaluated npArray of an image, to_color switch
#    Output: resultant npArray of an image, color plot method
#    Side effects: none
#    Purposes: produce resultant image npArray(weight distribution or original image) and color plot method from manipulating original image npArray
def  transChannel(imgDataEval,to_color=0):
    if np.min(imgDataEval) < 0 or to_color == 1:
        matR = ((np.uint8)(imgDataEval<0)) * imgDataEval * -1      
        matB = ((np.uint8)(imgDataEval>0)) * imgDataEval      
        img = np.zeros((matR.shape[0],matR.shape[1],3),matR.dtype)
        img[:,:,0] = matR
        img[:,:,2] = matB
        # normalizing
        Min = np.min(img); Max = np.max(img);
        img = (img - Min) / (Max - Min)       
        # cast to uint8
        img = tf.image.convert_image_dtype(img, tf.uint8)
        return img.eval(), None
    else:
        img = tf.image.convert_image_dtype(imgDataEval, tf.uint8)
        return img.eval(), plt.cm.gray

# pltShowImg(imgDataEval,channel=1,to_color=0):
#    Type: npArray, int, int ==> void
#    Input: evaluated npArray of an image, channel number, to_color switch
#    Output: none
#    Side effects: plot image
#    Purposes: plot and display image based on image npArray
def  pltShowImg(imgDataEval,channel=1,to_color=0): 
    if channel == 1: 
        imgComposedData = transChannel(imgDataEval,to_color)
        plt.imshow(imgComposedData[0],imgComposedData[1]) 
    else:
        plt.imshow(imgDataEval)
    return
     
#------------------------------Image Plot Scheme Functions------------------------------#