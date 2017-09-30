import os
import sys
import numpy as np
import keras
# Keras Imports
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Merge
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.merge import Add,Concatenate
from keras.layers.recurrent import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
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
image_embedding_size=4096

def image_embedding(input_tensor,mtype='inception'):
    '''
    Loads the image model.
    We have three options: VGG16, Resnet, InceptionV3
    args:
        mtype: {vgg,resnet,inception}, Default: 'inception'. Used for indicating the type of image model
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
    base_model=cnn_model(weights='imagenet',input_tensor=input_tensor)
    if mtype=='vgg':
        base_model.layers.pop()
        base_model.layers.pop()
    else:
        base_model.layers.pop()

    image_embedding_size=base_model.layers[-1].output._keras_shape[1]
    # Freeze the training of layers
    for layer in base_model.layers:
        layer.trainable = False

    # Will phase this out to Image model, which will take these embeddings and get them into the word dimension space
    # output=Dense(embedding_dim*2,activation='relu')(base_model.output)
    # output=Dense(embedding_dim,activation='relu')(output)

    # Give the summary of image model
    # model.summary()



def main_model():
    '''
    Will concatenate the two models. And do stuff.
    '''
    #The trainable image model. Takes the image embedding as input
    im_model = Sequential()
    im_model.add(Dense(embedding_dim*2,
                       activation='relu',
                       input_shape=(image_embedding_size,)))

    im_model.add(Dense(embedding_dim,
                       activation='relu'))

    im_model.add(RepeatVector(max_caption_length))
    # print im_model.input.shape
    # im_model.summary()

    lg_model = Sequential()
    # add Embedding layer
    lg_model.add(Embedding(vocab_size,
                           embedding_dim,
                           input_length=max_caption_length))

    # The output of the GRU layer is: [Batch_Size, TimeSteps, Recur_Output_dim]
    lg_model.add(GRU(recur_output_dim,return_sequences=True))

    '''
    Each of the Dense unit in the TimeDistributed Layer is connected to each of the recur_output_dim
    Add Time Distributed Layer, which will help in the given many-to-many task
    The output of the Time Distributed layer is: [Batch_Size, TimeSteps, embedding_dim]
    '''
    lg_model.add(TimeDistributed(Dense(embedding_dim)))
    # lg_model.summary()

    # Concatenates the image and sentence embedding
    merged_input = keras.layers.concatenate([im_model.output,lg_model.output],axis=-1)
    lstm_output = LSTM(1000)(merged_input)
    output = Dense(vocab_size,activation='softmax')(lstm_output)
    model=Model(inputs=[im_model.input,lg_model.input],outputs=output)
    model.summary()


# Just put for testing
# model_image(Input(shape=input_shape),'inception')
# model_language()
main_model()
