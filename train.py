import os
import sys
import cPickle as pickle
from model import main_model
from datagen import dataFeeder
from keras.models import Sequential, load_model


def train():
	params={
	'GLOVE_DIR': '/data/'+str(sys.argv[1])+'/keras-captioning/files/GLOVE',
	'EMBEDDING_DIM': 50,
	'MAX_SEQUENCE_LENGTH': 15,
	'VOCAB_SIZE': 7705,
	'RECUR_OUTPUT_DIM': 512,
	'IMAGE_ENCODING_SIZE': 4096,
	'PATH_TRAIN': '/data/'+str(sys.argv[1])+'/keras-captioning/files/Flickr8k_text/flickr_8k_train_dataset.txt',
	'PICKLE_FILE': '/data/'+str(sys.argv[1])+'/keras-captioning/files/encoded_images.p',
	'SPE': 20,
	'EPOCHS': 20,
	'BATCH_SIZE': 64,
	'SAVE_PATH': '/data/'+str(sys.argv[1])+'/keras-captioning/files/models/'
	}
	
	name = 'testing_.h5'

	# Get the model from main_model
	main = main_model(params)
	model = main.model

	# Get the generator from dF
	generator = main.gen
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit_generator(generator,verbose=2,epochs=params['EPOCHS'],steps_per_epoch=params['SPE'])
	model.fit_generator(generator,verbose=1,epochs=params['EPOCHS'],steps_per_epoch=params['SPE'])

	model.save(params['SAVE_PATH']+name)
	

if __name__=='__main__':
	train()
