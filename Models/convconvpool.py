import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Layer, Dense, Activation, Flatten, Input, Add, Reshape, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import pickle
import keras.backend as K
from keras import optimizers
from keras.applications.resnet import ResNet50
import time

class Convtrain(object):
	''' This class contains the convolutional model containing a vgg-like 
	conv-conv-batchnorm-pool architecture to train a classifier on the spectrograms. '''

	def __init__(self, input_shape = (200,1024,3), optimizer = 'Adam',
		loss='categorical_crossentropy', metrics=['accuracy']):

		self.input_shape = input_shape
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics




		self.model = Sequential()

		self.model.add(Convolution2D(32, kernel_size=2, strides=2, activation='relu', input_shape=self.input_shape))
		self.model.add(Convolution2D(32, kernel_size=2, strides=2, activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		self.model.add(Convolution2D(64, kernel_size=2, strides=1, activation='relu'))
		self.model.add(Convolution2D(64, kernel_size=2, strides=1, activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


		self.model.add(Dropout(0.5))
		self.model.add(Flatten())
		self.model.add(Dense(1024, ))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(8, activation='softmax'))

	def compile(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

	def fit(self, X, y, batch_size=32, epochs=100, verbose=1, callbacks=None,validation_split=0.2,
                validation_data=None, shuffle=True,
                initial_epoch=0):
		print("Training model:")
		start_time = time.time()
		history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, 
				validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, initial_epoch=initial_epoch)
        
		end_time = time.time()
		train_time = end_time - start_time
		#K.clear_session()

		print("Training finished!")
		print("Training time taken: {}s".format(train_time))
		return history
	
	def fit_generator(self, train_generator, validation_generator, epochs=20, steps_per_epoch=319):
		print("Training model:")
		start_time = time.time()
		history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_generator, validation_steps=40)
		end_time = time.time()
		train_time = end_time - start_time
		print("Training finished!")
		print("Time taken: {}s".format(train_time))
		return history

	def predict_generator(self, generator,steps=10, max_queue_size=10, workers=1):
		print("Predicting!")
		predictions = self.model.predict_generator(self, generator, steps, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=True, verbose=0)
		return predictions



	def predict(self,X):
		self.predictions = self.model.predict(X)
		return self.predictions

	def save(self,filepath):
                #save the keras model
		self.model.save(filepath+'model'+ '.h5')

	def load(self,filepath):
		self.model = load_model(filepath+'model'+ '.h5')


