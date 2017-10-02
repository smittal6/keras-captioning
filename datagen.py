import os
import sys
import numpy as np
import cv2
from keras.layers import Input
from embed_image import image_module

class DataGen():

    def __init__(self,list_path,batch_size = 32):
        '''
        Initializes the data module, gives the image embeddings and the word input in required form
        '''
        self.batch_size = batch_size
        self.local_list=load(self.list_path,'r').split() #Not sure about splitting.
        self.list_path = list_path

    def feedData(self):
        '''
        #This is the generator
        Given the list of image file names, gets them, converts to embedding.
        For the captions: convert them to unique index numbering(through call to a different function)
        marks the relevant positions with 1, leaves others as zero, in the vocab_length vector
        '''
        image_list = sampler()
        images = [] #To store the images as numpy arrays
        for img_path in image_list:
            img = load_image(img_path)
            images.append(img)
        images = np.asarray(images)

    def load_image(self,img_path):
        img = cv2.imread(img_path)
        # Do normalization if required
        return img

    def sampler(self):
        '''
        From the list of images, sample few according to batch size.
        Return the corresponding captions for each image
        So the captions list will be like this:
            captions[i] contains m number of captions for i^th image
        '''

