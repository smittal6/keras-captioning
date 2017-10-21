import os
import time
import numpy as np
import cPickle as pickle
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Embedding
from keras.models import load_model

class dataFeeder():

    def getVec(self,text):
        '''
        Returns the embedding for the sentence.
        '''
        sequence = [item for sublist in self.tokenizer.texts_to_sequences(text.split()) for item in sublist]
        valid_sequence = []
        for s in sequence:
        	if (s < self.params['VOCAB_SIZE']):
        		valid_sequence.append(s)
        padded_sequence = np.pad(sequence, (0,self.params['EMBEDDING_DIM'] - len(valid_sequence)%self.params['EMBEDDING_DIM']), 'constant')
        return padded_sequence

    def getHotVec(self,word):
        '''
        Returns the many hot vector as required by output
        '''
        x = np.zeros(self.params['VOCAB_SIZE'])
	if (word in self.word_index):
            	x[self.word_index[word]] = 1
        return x

    def sample(self):
        while 1:
        	img_list = []
        	encode_list = []
        	p1_embed_list = []
        	p2_hot_list = []
        	with open(self.params['PATH_TRAIN']) as f:
        		lines = random.sample(f.readlines(),self.params['BATCH_SIZE'])
    		for i,line in enumerate(lines):
    			lines[i] = lines[i].replace("\n","")
    			L = lines[i].split("\t")
                	img_list.append(L[0])
                	cap = L[1].split()
			encode_list.append(self.encoding[L[0]])
			p1_embed_list.append(self.getVec(' '.join(cap)))
			M = np.zeros((self.params['MAX_SEQUENCE_LENGTH'],self.params['VOCAB_SIZE']));
                        # print "Shape of Target: ",M.shape
                        caption_ind = 1
			for ind in range(0,len(cap)-1):
                            M[ind,:] = self.getHotVec(cap[caption_ind])
                            caption_ind += 1
                        # print "The One-Hot Matrix: ",M
			p2_hot_list.append(M)
        	inputs = [np.asarray(encode_list),np.asarray(p1_embed_list)]
        	output = np.asarray(p2_hot_list)
		# print 'input_shape = ', inputs[0].shape, inputs[1].shape
		# print 'output_shape = ', output.shape
    		yield (inputs, output)

    def __init__(self, params, picklefile, modelfile=None):
        texts = []
        with open(params['PATH_TRAIN']) as inf:
            for line in inf:
                temp = line.replace("\n","")
                texts.append(temp[temp.index('\t')+1:])

        tokenizer = Tokenizer(num_words = params['VOCAB_SIZE'], filters='!"$%&()*+,-/:;<=>?@[\\]^_`{|}~\n')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        z = tokenizer.word_index
        word_index = {}
        for word in z:
        	if (z[word] < params['VOCAB_SIZE']):
        		word_index[word] = z[word]
        pickle.dump(word_index, open("word_pickle.p", "wb" ))

        self.tokenizer = tokenizer
        self.word_index = word_index

        embeddings_index = {}
        f = open(os.path.join(params['GLOVE_DIR'], 'glove.6B.50d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        # Obtained all the word embeddings. Fetch for word 'w', as embeddings_index[w]

        embedding_matrix = np.zeros((params['VOCAB_SIZE'] + 1, params['EMBEDDING_DIM']))
        for word, i in word_index.items():
        	if (i < params['VOCAB_SIZE']):
	            embedding_vector = embeddings_index.get(word)
	            if embedding_vector is not None:
	                # words not found in embedding index will be all-zeros.
	                embedding_matrix[i] = embedding_vector
        #Found the intersection of GLOVE, and Captions

        # Initializing the class params like model, encoding_dict 
        self.encoding=pickle.load(open(picklefile,'rb'))
        self.word_index=word_index
        self.embedding_matrix = embedding_matrix
        self.params = params

        if (modelfile == None):
            self.model = Sequential()
            embedding_layer = Embedding(params['VOCAB_SIZE'] + 1,  params['EMBEDDING_DIM'], weights=[embedding_matrix], input_length=params['MAX_SEQUENCE_LENGTH'], trainable=False)
            self.model.add(embedding_layer)
        else:
            self.model = load_model(modelfile);

