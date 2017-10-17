import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_image(path):
    x = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(path, target_size=(224,224,3))), axis=0))
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
f = open(os.getcwd()+'/files/GLOVE/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

path = sys.argv[1]
encoding_model = get_model()
enc = get_encoding(encoding_model,path)
word_index = pickle.load(open('word_pickle.p','rb'))
rev_word_index = {v: k for k, v in word_index.iteritems()}

emb = np.zeros(50)

model = load_model(os.getcwd()+'/files/models/testing.h5')
enc = np.reshape(enc,(1,enc.shape[0]))
emb = np.reshape(emb,(1,emb.shape[0]))
emb[0][0] = word_index['#']

k = 50
s = ''
for i in range (1,k):
	wordvec = model.predict([enc,emb]).flatten()
	max_ind = np.argmax(wordvec)
	emb[0][i] = max_ind
	s = s + rev_word_index[max_ind]
	# if (rev_word_index[max_ind] == '.'):
	#	break
print s

emb = np.zeros(50)
emb = np.reshape(emb,(1,emb.shape[0]))
wordvec = model.predict([enc,emb]).flatten()
top_k_ind = (-wordvec).argsort()[:k]
print ''
for i in range (0,k):
	print rev_word_index[top_k_ind[i]],
print ''
