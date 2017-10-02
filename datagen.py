import os
import sys
import numpy as np
import cv2

class DataGen():

    def __init__(self,batch_size = 32):
        '''
        Initializes the data module, gives the image embeddings and the word input in required form
        '''
        self.batch_size = batch_size

    def feedData():
        '''
        Given the list of image file names, gets them, converts to embedding.
        For the image captions: convert them to unique index numbering
        marks the relevant positions with 1, leaves others as zero, in the vocab_length vector
        '''
