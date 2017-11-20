import os
import sys
from model import main_model
from datagen import dataFeeder
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint


def train():
	params={
	'GLOVE_DIR': '/data/'+str(sys.argv[1])+'/keras-captioning/files/GLOVE',
	'EMBEDDING_DIM': 128,
	'MAX_SEQUENCE_LENGTH': 20,
	'VOCAB_SIZE': 7706,
	'RECUR_OUTPUT_DIM': 512,
	'IMAGE_ENCODING_SIZE': 4096,
	'PATH_TRAIN': '/data/'+str(sys.argv[1])+'/keras-captioning/files/Flickr8k_text/flickr_8k_train_dataset.txt',
	'PICKLE_FILE': '/data/'+str(sys.argv[1])+'/keras-captioning/files/encoded_images.p',
	'SPE': 128,
	'EPOCHS': 100,
	'BATCH_SIZE': 128,
	'SAVE_PATH': '/data/'+str(sys.argv[1])+'/keras-captioning/files/models/'
	}
	

	# Get the model from main_model
	main = main_model(params)
	model = main.model

	# Get the generator from dF
	generator = main.gen
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	newpath = params['SAVE_PATH'] + sys.argv[2]
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	filepath = params['SAVE_PATH'] + sys.argv[2] + "/weights-improvement-{epoch:02d}.hdf5"
	checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]

	model.fit_generator(generator,verbose=1,epochs=params['EPOCHS'],steps_per_epoch=params['SPE'],callbacks=callbacks_list)
	name = 'abc.hd5'
	model.save(params['SAVE_PATH'] + name)
	

if __name__=='__main__':
	train()
