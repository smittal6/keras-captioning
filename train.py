import os
import sys
from model import main_model
from datagen import dataFeeder
from keras.models import Sequential, load_model

def train():
	params={
	'GLOVE_DIR':'',
	'EMBEDDING_DIM': 50,
	'MAX_SEQUENCE_LENGTH': 50,
	'VOCAB_SIZE':20000,
	'RECUR_OUTPUT_DIM':100,
	'IMAGE_ENCODING_SIZE':4096,
	'PATH_TRAIN':''
	'PICKLE_FILE': "encoded_image.p"
	'SPE':10,
	'EPOCHS':15,
	'SAVE_PATH':''
	}
	
	name='glove_gru_100recur_vgg.h5'
	# Get the model from main_model
	main=main_model()
	model=main.model

	# Get the generator from dF
	generator=main.gen

	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit_generator(generator,verbose=2,epochs=params['EPOCHS'],steps_per_epoch=params['SPE'])

	model.save(params['SAVE_PATH']+name)
	

if __name__=='__main__':
	train()