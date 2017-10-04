import os
import time
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

GLOVE_DIR = os.getcwd()+'/GLOVE/'
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000

texts = []
with open(os.getcwd()+'/Flickr8k_text/Flickr8k.token.txt') as inf:
    for line in inf:
        texts.append(line[line.index('#')+3:-2])

print 'Found %s texts.' % len(texts)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print 'Found %s unique tokens.' % len(word_index)
print 'Shape of data tensor:', data.shape

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print 'Found %s word vectors.' % len(embeddings_index)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,  EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

print embedding_layer