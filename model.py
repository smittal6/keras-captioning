import os
import sys
import numpy as np

# Keras Imports
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

def model_image(mtype='vgg',input_tensor):
    '''
    Loads the image model.
    We have three options: VGG16, Resnet, InceptionV3
    args:
        mtype: {vgg,resnet,inception}, Default: 'vgg'. Used for indicating the type of image model
    '''
    if mtype='resnet':
        # One FC at Top
        from keras.applications.resnet50 import ResNet50 as cnn_model
    if mtype='inception':
        # One FC at Top
        from keras.applications.inception_v3 import InceptionV3 as cnn_model
    if mtype='vgg':
        # VGG has three FC layers at the top
        from keras.applications.vgg16 import VGG16 as cnn_model

    # Load the pretrained weights(on imagenet)
    # Not including the top layer as it is tuned for class pred, and output of convolution layers is average pooled
    base_model=cnn_model(weights='imagenet',input_tensor=input_tensor,include_top=False,pooling='avg')

# Just put for testing
model_image('vgg',Input(shape=(224,224,3)))


