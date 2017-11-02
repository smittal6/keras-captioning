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
from keras import regularizers
from keras.utils import plot_model
from datagen import dataFeeder

class main_model():
    def __init__  (self, params):
        '''
        Takes the input as *image embedding* and vector like output form described below.
        Gives the output probabilities of various words from the vocab, given the image and previous words.
        '''
        #The trainable image model. Takes the image embedding as input
        im_model = Sequential()
        im_model.add(Dense(params['EMBEDDING_DIM']*2, activation='relu', input_shape=(params['IMAGE_ENCODING_SIZE'],)))
        im_model.add(Dense(params['EMBEDDING_DIM'], activation='relu'))
        im_model.add(RepeatVector(params['MAX_SEQUENCE_LENGTH']))
        # print im_model.input.shape
        # im_model.summary()

        dF = dataFeeder(params,picklefile=params['PICKLE_FILE'])
        lg_model = dF.model
        # lg_model.summary()
        # We have obtained the word embeddings here, inherently with dF.model

        # Concatenates the image and sentence embedding
        merged_input = keras.layers.concatenate([im_model.output,lg_model.output],axis=-1)
        # print merged_input._keras_shape
        recurrent_layer = LSTM(params['RECUR_OUTPUT_DIM'],return_sequences=False,kernel_regularizer = regularizers.l2(),recurrent_regularizer = regularizers.l2())(merged_input)
        # output = TimeDistributed(Dense(params['VOCAB_SIZE'],activation='softmax',kernel_regularizer = regularizers.l2()))(recurrent_layer)
        output = Dense(params['VOCAB_SIZE'],activation='softmax',kernel_regularizer = regularizers.l2())(recurrent_layer)

        # Defining the functional model
        model = Model(inputs=[im_model.input,lg_model.input],outputs=output)
        self.model = model
        g = dF.sample()
        self.gen = g

        # Plot the model
        # plot_model(model,'try1.png',show_shapes=True)
