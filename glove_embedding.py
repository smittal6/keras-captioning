import os
import time
import numpy as np
import cPickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import load_model

GLOVE_DIR = os.getcwd()+'/GLOVE/'
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
PATH_TRAIN = ''

class dataFeeder():
    '''
    Returns the embedding for the given sentence
    '''
    def __init__(self,picklefile,modelfile=None):
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
        # Obtained all the word embeddings. Fetch for word 'w', as embeddings_index[w]

        # print 'Found %s word vectors.' % len(embeddings_index)

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        #Found the intersection of Glove, and Captions

        # Initializing the class params like model, encoding_dict 
        self.encoding=pickle.load(open(picklefile,'rb'))
        self.word_index=word_index
        self.embedding_matrix = embedding_matrix

        if modelfile==None:
            self.model = Sequential()
            embedding_layer = Embedding(len(word_index) + 1,  EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
            self.model.add(embedding_layer)
        else:
            self.model = load_model(modelfile);

    def getVec(self,text):
        '''
        Returns the embedding for the sentence.
        '''
        sequence=[item for sublist in tokenizer.texts_to_sequences(text) for item in sublist]
        b=np.pad(sequence, (0,EMBEDDING_DIM - len(sequence)%EMBEDDING_DIM), 'constant')
        return self.model.predict(np.reshape(b,(1,b.shape[0])))

    def getHotVec(self,text):
        '''
        Returns the many hot vector as required by output
        '''
        x = np.zeros(MAX_NB_WORDS)
        words = text.split()
        for word in words:
                if (word in self.word_index):
                        x[self.word_index[word]] = 1
        return x

    def sample(self, batch_size = 32):
        '''
        Takes as input batch size
        Sends: Encoded Image, Glove embedding (p1), ManyHotVec (p2)
        '''
	img_list = []
        encode_list = []
        p1_embed_list = []
        p2_hot_list = []
        with open(os.getcwd()+PATH_TRAIN) as f:
                lines = random.sample(f.readlines(),batch_size)

        for i,line in enumerate(lines):
                lines[i] = lines[i].replace("\n","")
                L = lines[i].split("\t")
                img_list.append(L[0])
                encode_list.append(self.encoding[L[0]])
                cap = L[1].split()
                ind = random.randint(1,len(cap)-1)
                p1_embed_list.append(getVec(' '.join(cap[:ind]))
                p2_hot_list.append(getHotVec(' '.join(cap[ind:])))
        return zip(encode_list,p1_embed_list,p2_hot_list)


