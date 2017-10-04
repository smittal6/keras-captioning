import os
import time
import numpy as np
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

GLOVE_DIR = os.getcwd()+'/GLOVE/'
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
D = '/Flickr8k_text/flickr_8k_train_dataset.txt'

def getVec (text, model):
    sequence=[item for sublist in tokenizer.texts_to_sequences(text) for item in sublist]
    b=np.pad(sequence, (0,EMBEDDING_DIM - len(sequence)%EMBEDDING_DIM), 'constant')
    return model.predict(np.reshape(b,(1,b.shape[0])))

def getHotVec (text, word_index):
	x = np.zeros(MAX_NB_WORDS)
	words = text.split()
	for word in words:
		if (word in word_index):
			x[word_index[word]] = 1
	return x

def sample (path, batch_size, model, word_index):
	img_list = []
	p1_embed_list = []
	p2_list = []
	with open(os.getcwd()+path) as f:
		lines = random.sample(f.readlines(),batch_size)
	for i,line in enumerate(lines):
		lines[i] = lines[i].replace("\n","")
		L = lines[i].split("\t")
		img_list.append(L[0])
		cap = L[1].split()
		ind = random.randint(1,len(cap)-1)
		p1_embed_list.append(getVec(' '.join(cap[:ind]),model))
		p2_list.append(getHotVec(' '.join(cap[ind:]),word_index))
	return zip(img_list,p1_embed_list,p2_list)

texts = []
with open(os.getcwd()+'/Flickr8k_text/Flickr8k.token.txt') as inf:
    for line in inf:
        texts.append(line[line.index('#')+3:-1])

tokenizer = Tokenizer(num_words = MAX_NB_WORDS, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
model = Sequential()
model.add(embedding_layer)

S = sample(D,2,model,word_index)

print S