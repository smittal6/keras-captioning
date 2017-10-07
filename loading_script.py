import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

def load_image(path):
    x = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(path, target_size=(224,224))), axis=0))
    return np.asarray(x)

def get_model():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	model.layers.pop()
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []
	return model

def get_encoding(model, path):
	image = load_image(path)
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	return pred

embeddings_index = {}
f = open('/data/rohitkb/keras-captioning/files/GLOVE/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# path = os.getcwd()+'/Flickr8k_Dataset/Flickr8k_Dataset/'+'311146855_0b65fdb169.jpg'
path = 'climb.jpg'
encoding_model = get_model()
enc = get_encoding(encoding_model,path)
word_index = pickle.load(open('word_pickle.p','rb'))
rev_word_index = {v: k for k, v in word_index.iteritems()}
emb = embeddings_index['#']

# model = load_model('glove_gru_100recur_vgg.h5')
model = load_model('/data/rohitkb/keras-captioning/models/glove_gru_100recur_vgg.h5')
enc = np.reshape(enc,(1,enc.shape[0]))
emb = np.reshape(emb,(1,emb.shape[0]))
wordvec = model.predict([enc,emb])