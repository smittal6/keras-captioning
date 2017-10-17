import os
import sys
import cPickle as pickle
from model import main_model
from datagen import dataFeeder
from keras.models import Sequential, load_model

def train():
	params={
	'GLOVE_DIR':'/home/rohitkb/Desktop/Coursework/CS771/Project/GLOVE',
	'EMBEDDING_DIM': 50,
	'MAX_SEQUENCE_LENGTH': 50,
	'VOCAB_SIZE': 5000,
	'RECUR_OUTPUT_DIM':100,
	'IMAGE_ENCODING_SIZE':4096,
	'PATH_TRAIN':'/home/rohitkb/Desktop/Coursework/CS771/Project/Flickr8k_text/flickr_8k_train_dataset.txt',
	'PICKLE_FILE': "/home/rohitkb/Desktop/Coursework/CS771/Project/encoded_images.p",
	'SPE':10,
	'EPOCHS':15,
	'BATCH_SIZE':64,
	'SAVE_PATH':'/home/rohitkb/Desktop/Coursework/CS771/Project/models/'
	}
	
	name = 'faltu.h5'

	# Get the model from main_model
	main = main_model(params)
	model = main.model

	# Get the generator from dF
	generator = main.gen

	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit_generator(generator,verbose=2,epochs=params['EPOCHS'],steps_per_epoch=params['SPE'])

	model.save(params['SAVE_PATH']+name)
	

if __name__=='__main__':
	train()
