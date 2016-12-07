import os, sys, shutil, time
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano
theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'
import keras
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.regularizers import l2, activity_l1l2, activity_l1, activity_l2
from keras.layers.core import Dense, Lambda, Activation, ActivityRegularization, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge, merge, Input
from keras.models import Model
from keras.engine import Layer
from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
from keras.constraints import nonneg
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#print X_train.shape  # (50000, 3, 32, 32)
#print y_train        # (50000, 1)            [[0], [1], [9]]  

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test  /= 255

y_train = np_utils.to_categorical(y_train, 10).astype('float32')
y_test  = np_utils.to_categorical(y_test,  10).astype('float32')

BATCH = 100
LATENT = 150
H,W = X_train.shape[-2:]

class Gan:
	def __init__(self):
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')
		b1    = BatchNormalization(mode=0, axis=1)
		conv2 = Convolution2D(64, 3, 3, activation='relu', subsample=(2,2), border_mode='same')
		b2    = BatchNormalization(mode=0, axis=1)
		conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
		b3    = BatchNormalization(mode=0, axis=1)
		conv4 = Convolution2D(128, 3, 3, activation='relu', subsample=(2,2), border_mode='same')
		b4    = BatchNormalization(mode=0, axis=1)
		conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')
		b5    = BatchNormalization(mode=0, axis=1)
		conv6 = Convolution2D(256, 3, 3, activation='relu', subsample=(2,2), border_mode='same')
		b6    = BatchNormalization(mode=0, axis=1)
		flat = Flatten()
		d1 = Dense(256, activation='relu')
		d2 = Dense(10, activation='softmax')

		inp = Input( batch_shape=(None,3,H,W), name="inp_train" )
		z = b1(conv1(inp))
		z = b2(conv2(z))
		z = b3(conv3(z))
		z = b4(conv4(z))
		z = b5(conv5(z))
		z = b6(conv6(z))
		z = d1(flat(z))
		class1 = d2(z)

		lat_dense1 = Dense(256, activation='relu')
		lat_dense1_bn = BatchNormalization(mode=0, axis=1)
		lat_dense2 = Dense(256*W/8*H/8, activation='relu')
		lat_dense2_bn = BatchNormalization(mode=0, axis=1)
		lat_reshape = Reshape( (256,H/8,W/8) )
		dec1 = Deconvolution2D(128, 3, 3, output_shape=(None,128,H/4,W/4), subsample=(2,2), border_mode='same')
		dec1_bn = BatchNormalization(mode=0, axis=1)
		dec2 = Deconvolution2D(64, 3, 3, output_shape=(None,64,H/2,W/2), subsample=(2,2), border_mode='same')
		dec2_bn = BatchNormalization(mode=0, axis=1)
		dec3 = Deconvolution2D(32, 3, 3, output_shape=(None,32,H,W), subsample=(2,2), border_mode='same')
		dec3_bn = BatchNormalization(mode=0, axis=1)
		dec4 = Deconvolution2D(3, 3, 3, output_shape=(None,3,H,W), border_mode='same')

		latent = Input( batch_shape=(None,LATENT), name="latent" )
		g = lat_dense1_bn(lat_dense1(latent))
		g = lat_dense2_bn(lat_dense2(g))
		g = lat_reshape(g)
		g = dec1_bn(dec1(g))
		g = dec2_bn(dec2(g))
		g = dec3_bn(dec3(g))
		g = dec4(g)

		self.class_model = Model( input=inp, output=class1 )
		def modified_categorical_crossentropy(y_true, y_pred):
			y_pred *= 0.9
			_EPSILON = K.epsilon()
			return K.mean(K.sum(-y_true*K.log(K.clip(y_pred, _EPSILON, 1-_EPSILON)), axis=-1))
		self.class_model.compile(
			optimizer=Adam(lr=0.002, beta_2=0.999),
			loss=modified_categorical_crossentropy,
			metrics=['accuracy'])

		fake_merged_with_real = merge( [inp,g], mode='concat', concat_axis=0 )

		self.discriminator_model = Model( input=fake_merged_with_real, output=class1 )
		self.discriminator_model.compile(
			optimizer=Adam(lr=0.002, beta_2=0.999),
			loss=modified_categorical_crossentropy,
			metrics=['accuracy'])

		for x in [conv1,b1,conv2,b2,conv3,b3,conv4,b4,conv5,b5,conv6,b6,d1,d2]:
			x.trainable = False   # should do it before compiling generator

		self.generator_fool_model = Model( input=latent, output=g )
		self.generator_fool_model.compile(
			optimizer=Adam(lr=0.002, beta_2=0.999),
			loss=modified_categorical_crossentropy,
			metrics=['accuracy'])

gan = Gan()
gan.class_model.summary()

import scipy
import scipy.misc
import scipy.misc.pilutil

def batch_to_jpeg(batch, fn="test.png"):
	BATCH = batch.shape[0]
	W = batch.shape[-1]
	H = batch.shape[-2]
	side = int( np.ceil(np.sqrt(BATCH)) )
	pic = np.zeros( shape=(3,side*H,side*W) )
	for y in range(side):
		for x in range(side):
			i = y*side + x
			if i > BATCH: break
			pic[:, y*H:(y+1)*H, x*W:(x+1)*W] = batch[i]
	scipy.misc.toimage(pic, cmin=0.0, cmax=1.0).save(fn)

cursor = 1e10
epoch = 0
while 1:
	if cursor+BATCH > X_train.shape[0]:
		sh = np.random.choice( X_train.shape[0], size=X_train.shape[0], replace=False )
		shuffled_x_train = X_train[sh]
		shuffled_y_train = y_train[sh]
		cursor = 0
		epoch += 1
		loss, acc = gan.class_model.evaluate( X_test, y_test, batch_size=BATCH)
		print "e%i test ---- loss=%0.3f acc=%0.2f%%" % (epoch, loss, acc*100)

	x = shuffled_x_train[cursor:cursor+BATCH]
	y = shuffled_y_train[cursor:cursor+BATCH]

	loss, acc = gan.class_model.train_on_batch( x, y )
	if cursor % 10000 == 0:
		print "e%i:%05i loss=%0.3f acc=%0.2f%%" % (epoch, cursor, loss, acc*100)
		batch_to_jpeg( shuffled_x_train[:BATCH] )

	cursor += BATCH

