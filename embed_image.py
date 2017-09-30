import sys
import os
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

class image_module():
    '''
    Load the complete model, and then pop the layer.
    This is better as compared to loading only the top as we won't have to train the FC layer
    '''
    def __init__(self,input_vector,modeltype='inception'):

        if mtype == 'resnet':
            # One FC at Top
            model = ResNet50(weights = 'imagenet',input_tensor = input_vector)
            model.layers.pop()

        if mtype == 'inception':
            # One FC at Top
            model = InceptionV3(weights = 'imagenet',input_tensor = input_vector)
            model.layers.pop()

        if mtype == 'vgg':
            # VGG has three FC layers at the top
            model = VGG16(weights = 'imagenet',input_tensor = input_vector)
            model.layers.pop()
            model.layers.pop()

        for layers in model.layers:
            layer.trainable = False;

        self.model = model
        self.modeltype = modeltype

    def get_embedding(image_input):
        '''
        Helper function to get the image embedding, from the chosen model
        '''
        return model.predict(image_input)
