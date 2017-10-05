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
    input_shape,batch_size, max_caption_length, vocab_size, embedding_dim
'''

#Temp definitions for testing: 
vocab_size = 20000
max_caption_length = 50 # 'Time Samples'/Time step
embedding_dim = 50 # Each vector will be represented by these many dimensions
recur_output_dim = 100 # Dimensionality of output space of GRU
input_shape = (224,224,3)
image_embedding_size = 4096

class main_model():
    def __init__  (self):
        '''
        Takes the input as *image embedding* and vector like output form described below.
        Gives the output probabilities of various words from the vocab, given the image and previous words.
        Doesn't seem to be appropriate for testing, as we need to sample there.
        '''
        #The trainable image model. Takes the image embedding as input
        im_model = Sequential()
        im_model.add(Dense(embedding_dim*2, activation='relu', input_shape=(image_embedding_size,)))
        im_model.add(Dense(embedding_dim, activation='relu'))
        im_model.add(RepeatVector(max_caption_length))
        # print im_model.input.shape
        # im_model.summary()

        dF = dataFeeder(picklefile='encoded_images.p')
        lg_model = dF.model
        g = dF.sample()
        # add Embedding layer
        # lg_model.add(Embedding(vocab_size, embedding_dim, input_length=max_caption_length))

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
        '''
        We have a choice now. return_sequences=True will give the progress of LSTM states, ie complete sequence.
        Applying Dense/TimeDistributedDense will lead to max_caption_length X vocab_size matrix, each row mapping the existence of a word
        Therefore, our output(the next words) that we feed should also be of the same format.
        The other option, take the final LSTM state, let it encode the information for the sentence.
        Map it with Dense Layer such that we have a vector of vocab_size, with 1's at places which have been marked as 'next words'
        For now, continuing with second option.
        Output form: 0 0 0 0 1 0 0 0 1 0 0 0 1 ..., 1 at ith place indicating presence of i^th word.
        '''
        lstm_output = LSTM(1000,return_sequences=False)(merged_input)
        output = Dense(vocab_size,activation='softmax')(lstm_output)
        model = Model(inputs=[im_model.input,lg_model.input],outputs=output)
        self.model = model
        self.gen = g

        #Plot the model
        # plot_model(model,'try1.png',show_shapes=True)

