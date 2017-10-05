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
from datagen import dataFeeder
'''
We need to have a dictionary storing following stuff:
    input_shape,batch_size, MAX_SEQUENCE_LENGTH, vocab_size, embedding_dim
'''

#Temp definitions for testing:
EMBEDDING_DIM = 50
VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 50
RECUR_OUTPUT_DIM = 100
IMAGE_ENCODING_SIZE = 4096
PICKLE_FILE = 'encoded_images.p'

class main_model():
    def __init__  (self, params):
        '''
        Takes the input as *image embedding* and vector like output form described below.
        Gives the output probabilities of various words from the vocab, given the image and previous words.
        Doesn't seem to be appropriate for testing, as we need to sample there.
        '''
        #The trainable image model. Takes the image embedding as input
        im_model = Sequential()
        im_model.add(Dense(params['EMBEDDING_DIM']*2, activation='relu', input_shape=(params['IMAGE_ENCODING_SIZE'],)))
        im_model.add(Dense(params['EMBEDDING_DIM'], activation='relu'))
        im_model.add(RepeatVector(params['MAX_SEQUENCE_LENGTH']))
        # print im_model.input.shape
        # im_model.summary()

        dF = dataFeeder(picklefile=params['PICKLE_FILE'])
        lg_model = dF.model
        g = dF.sample()
        # add Embedding layer
        # lg_model.add(Embedding(vocab_size, embedding_dim, input_length=MAX_SEQUENCE_LENGTH))

        # The output of the GRU layer is: [Batch_Size, TimeSteps, Recur_Output_dim]
        lg_model.add(GRU(params['RECUR_OUTPUT_DIM'],return_sequences=True))

        '''
        Each of the Dense unit in the TimeDistributed Layer is connected to each of the recur_output_dim
        Add Time Distributed Layer, which will help in the given many-to-many task
        The output of the Time Distributed layer is: [Batch_Size, TimeSteps, embedding_dim]
        '''
        lg_model.add(TimeDistributed(Dense(params['EMBEDDING_DIM'])))
        # lg_model.summary()

        # Concatenates the image and sentence embedding
        merged_input = keras.layers.concatenate([im_model.output,lg_model.output],axis=-1)
        '''
        We have a choice now. return_sequences=True will give the progress of LSTM states, ie complete sequence.
        Applying Dense/TimeDistributedDense will lead to MAX_SEQUENCE_LENGTH X vocab_size matrix, each row mapping the existence of a word
        Therefore, our output(the next words) that we feed should also be of the same format.
        The other option, take the final LSTM state, let it encode the information for the sentence.
        Map it with Dense Layer such that we have a vector of vocab_size, with 1's at places which have been marked as 'next words'
        For now, continuing with second option.
        Output form: 0 0 0 0 1 0 0 0 1 0 0 0 1 ..., 1 at ith place indicating presence of i^th word.
        '''
        lstm_output = LSTM(1000,return_sequences=False)(merged_input)
        output = Dense(params['VOCAB_SIZE'],activation='softmax')(lstm_output)
        model = Model(inputs=[im_model.input,lg_model.input],outputs=output)
        self.model = model
        self.gen = g

        #Plot the model
        # plot_model(model,'try1.png',show_shapes=True)

