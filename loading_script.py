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
word_freq = pickle.load(open('word_freq.p','rb'))
rev_word_index = {v: k for k, v in word_index.iteritems()}

emb = np.zeros(50)

model = load_model(os.getcwd()+'/files/models/testing.h5')
enc = np.reshape(enc,(1,enc.shape[0]))
emb = np.reshape(emb,(1,emb.shape[0]))
emb[0][0] = word_index['#']; emb[0][1] = word_index['a'];
rev_word_index[0] = 'vv'

word_prob = {}
s = 0
m = 0
for t in word_freq:
	s += word_freq[t]
for t in word_freq:
	word_prob[t] = s/float(word_freq[t])
	if (word_prob[t] > m):
		m = word_prob[t]
# for t in word_prob:
# 	word_prob[t] = word_prob[t]/float(m)*1000
print word_prob['.']
print word_prob['sash']

ans = model.predict([enc,emb])
# for i in range (0,ans.shape[1]):
	#print 'ans[i] , argmax = ', ans[0,i], np.argmax(ans[0,i])
	# print ans[0,i].shape
	# print rev_word_index[np.argmax(ans[0,i,:])],
