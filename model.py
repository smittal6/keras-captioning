import os
import sys
import numpy as np

# Keras Imports
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
'''
We need to have a dictionary storing following stuff:
    input_shape,batch_size, max_caption_length, vocab_size, embedding_dim
'''

#Temp definitions for testing:
vocab_size=500
max_caption_length=100 # 'Time Samples'/Time step
embedding_dim=256 # Each vector will be represented by these many dimensions
recur_output_dim=128 # Dimensionality of output space of GRU
input_shape=(200,200,3)

def model_image(input_tensor,mtype='vgg'):
    '''
    Loads the image model.
    We have three options: VGG16, Resnet, InceptionV3
    args:
        mtype: {vgg,resnet,inception}, Default: 'vgg'. Used for indicating the type of image model
    '''
    if mtype=='resnet':
        # One FC at Top
        from keras.applications.resnet50 import ResNet50 as cnn_model
    if mtype=='inception':
        # One FC at Top
        from keras.applications.inception_v3 import InceptionV3 as cnn_model
    if mtype=='vgg':
        # VGG has three FC layers at the top
        from keras.applications.vgg16 import VGG16 as cnn_model

    # Load the pretrained weights(on imagenet)
    # Not including the top layer as it is tuned for class pred, and output of convolution layers is average pooled
    base_model=cnn_model(weights='imagenet',input_tensor=input_tensor,include_top=False,pooling='avg')

    # Freeze the training of layers
    for layer in base_model.layers:
        layer.trainable = False

    # Give the summary of image model
    output=Dense(embedding_dim*2,activation='relu')(base_model.output)
    output=Dense(embedding_dim,activation='relu')(output)

    model=Model(inputs=base_model.input,outputs=output)
    model.summary()

    return model

def model_language():


    base_lang=Sequential()
    # add Embedding layer
    base_lang.add(Embedding(vocab_size,embedding_dim,input_length=max_caption_length))

    # The output of the GRU layer is: [Batch_Size, TimeSteps, Recur_Output_dim]
    base_lang.add(GRU(recur_output_dim,return_sequences=True))

    # Add Time Distributed Layer, which will help in many-to-many task
    base_lang.add(TimeDistributed(Dense(embedding_dim)))
    # The output of the Time Distributed layer is: [Batch_Size, TimeSteps, embedding_dim]

    # base_lang.summary()

    return base_lang #When the time is right, we'll return

def main_model():
    '''
    Will concatenate the two models. And do stuff.
    '''
    im_model = model_image(Input(shape=input_shape))
    lg_model = model_language()




# Just put for testing
model_image(Input(shape=input_shape),'inception')
# model_language()
