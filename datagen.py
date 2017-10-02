import os
import sys
import numpy as np
import cv2
from keras.layers import Input
from embed_image import image_module

class DataGen():

    def __init__(self,img_list_path,cap_list_path,batch_size = 32):
        '''
        Initializes the data module, gives the image embeddings and the word input in required form
        '''
        self.batch_size = batch_size
        # self.local_list=load(self.list_path,'r').split() #Not sure about splitting.
        self.img_list = load(img_list_path,'r')
        self.cap_list = load(cap_list_path'r')
        # one to one mapping b/w img_list_path, cap_list_path
        self.embedder = image_module()

    def feedData(self):
        '''
        #This is the generator
        Given the list of image file names, gets them, converts to embedding.
        For the captions: convert them to unique index numbering(through call to a different function)
        marks the relevant positions with 1, leaves others as zero, in the vocab_length vector
        '''
        images, captions = sampleData()
        # images are numpy now, captions is a list. Both have same length
        image_embeddings = []
        for image in images:
            temp=self.embedder.model.predict(image)
            image_embeddings.append(temp)
        image_embeddings = np.asarray(image_embeddings)

    def sampleData(self):
        '''
        From the list of images, sample few according to batch size.
        Return the corresponding captions for each image
        So the captions list will be like this:
            captions[i] contains m number of captions for i^th image
        '''
        indexes = sampleIndex()
        image_list = self.img_list[indexes]
        caption_list = self.cap_list[indexes]
        for img_path in image_list:
            img = load_image(img_path)
            images.append(img)
        images = np.asarray(images)
        #Get the captions here[Captions is a list]
        return images, captions

    def sampleIndex(self):
        '''
        For sampling the indexes. Will be used by sampleData.
        The total length of indexes will be batch size
        '''
        indexes = np.arange(len(self.local_list))
        np.random.shuffle(indexes)
        return indexes[0:self.batch_size]

    def load_image(self,img_path):
        img = cv2.imread(img_path)
        # Do normalization if required
        return img
