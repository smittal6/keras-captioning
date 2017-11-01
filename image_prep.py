import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.backend.tensorflow_backend import set_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

counter = 0

def load_image(path):
    x = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(path, target_size=(224,224,3))), axis=0))
    return np.asarray(x)

def get_model():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	model.layers.pop()
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []
	return model

def get_encoding(model, img):
	global counter
	counter += 1
	image = load_image(os.getcwd()+'/Flickr8k_Dataset/Flickr8k_Dataset/'+str(img))
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	print "Encoding image: "+str(counter)
	print pred.shape
	return pred

def prepare_dataset():
	encoded_images = {}
	encoding_model = get_model()
	with open(os.getcwd()+'/Flickr8k_text/Flickr8k.token.txt') as inf:
		for line in inf:
			img = line[0:line.index('#')]
			encoded_images[img] = get_encoding(encoding_model, img)
	with open("encoded_images.p", "wb") as pickle_f:
		pickle.dump( encoded_images, pickle_f)  

prepare_dataset()
