import os
import sys
import numpy as np

# Keras Imports
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

def model_image(mtype='vgg'):
    '''
    Loads the image model.
    We have three options: VGG16, Resnet, InceptionV3
    '''
