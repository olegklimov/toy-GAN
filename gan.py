import os, sys, shutil, time
os.environ['THEANO_FLAGS'] = "device=gpu"
import theano
theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_run'
theano.config.nvcc.fastmath = True
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
import scipy
import scipy.misc
import scipy.misc.pilutil

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#print X_train.shape  # (50000, 3, 32, 32)
#print y_train        # (50000, 1)            [[0], [1], [9]]  

BATCH  = 50
LATENT = 150
H,W = X_train.shape[-2:]
LABELS = 10
labels_text = "airplane automobile bird cat deer dog frog horse ship truck FAKE".split()

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 128
X_train -= 1.0
X_test  /= 128
X_test  -= 1.0

y_train = np_utils.to_categorical(y_train, LABELS+1).astype('float32')
y_test  = np_utils.to_categorical(y_test,  LABELS+1).astype('float32')

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
		d2 = Dense(LABELS+1, activation='softmax')

		inp = Input( batch_shape=(None,3,H,W), name="inp_train" )
		z = b1(conv1(inp))
		z = b2(conv2(z))
		z = b3(conv3(z))
		z = b4(conv4(z))
		z = b5(conv5(z))
		z = b6(conv6(z))
		z = d1(flat(z))
		class1 = d2(z)

		self.class_model = Model( input=inp, output=class1 )
		def modified_categorical_crossentropy(y_true, y_pred):
			y_pred *= 0.9/y_pred.sum(axis=-1, keepdims=True)
			_EPSILON = K.epsilon()
			return K.mean(K.sum(-y_true*K.log(K.clip(y_pred, _EPSILON, 1-_EPSILON)), axis=-1))
		for x in self.class_model.layers:
			x.trainable = False   # should do it before compiling generator
		self.class_model.compile(
			optimizer=Adam(lr=0.002, beta_2=0.999),
			loss=modified_categorical_crossentropy,
			metrics=['accuracy'])
		self.class_model.train_on_batch( X_train[:1], y_train[:1] )

		lat_dense1 = Dense(256, activation='relu')
		lat_dense2 = Dense(256*W/8*H/8, activation='relu')
		lat_reshape = Reshape( (256,H/8,W/8) )
		dec1 = Deconvolution2D(128, 3, 3, activation='relu', output_shape=(None,128,H/4,W/4), subsample=(2,2), border_mode='same')
		dec1_bn = BatchNormalization(mode=0, axis=1)
		dec2 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None,64,H/2,W/2), subsample=(2,2), border_mode='same')
		dec2_bn = BatchNormalization(mode=0, axis=1)
		dec3 = Deconvolution2D(32, 3, 3, activation='relu', output_shape=(None,32,H,W), subsample=(2,2), border_mode='same')
		dec3_bn = BatchNormalization(mode=0, axis=1)
		dec4 = Deconvolution2D(3, 3, 3, activation='tanh', output_shape=(None,3,H,W), border_mode='same')

		latent = Input( batch_shape=(None,LATENT), name="latent" )
		g = lat_dense1(latent)
		g = lat_dense2(g)
		g = lat_reshape(g)
		g = dec1_bn(dec1(g))
		g = dec2_bn(dec2(g))
		g = dec3_bn(dec3(g))
		g = dec4(g)
		self.generator_only_model = Model( input=latent, output=g )
		self.generator_only_model.compile(
			optimizer=Adam(lr=0.002, beta_2=0.999),
			loss="mse")

		#z = b1(conv1(g))
		#z = b2(conv2(z))
		#z = b3(conv3(z))
		#z = b4(conv4(z))
		#z = b5(conv5(z))
		#z = b6(conv6(z))
		#z = d1(flat(z))
		#class_generator = d2(z)
		self.class_model.trainable = False
		class_generator = self.class_model(g)

		def modified_categorical_crossentropy2(y_true, y_pred):
			y_pred *= 0.9/y_pred.sum(axis=-1, keepdims=True)
			y_pred += 0.1
			_EPSILON = K.epsilon()
			return K.mean(K.sum(-y_true*K.log(K.clip(y_pred, _EPSILON, 1-_EPSILON)), axis=-1))
		self.generator_fool_model = Model( input=latent, output=class_generator )
		self.generator_fool_model.compile(
			optimizer=Adam(lr=0.001, beta_2=0.999),
			loss=modified_categorical_crossentropy2,
			metrics=['accuracy'])
		prob1 = self.class_model.predict(X_train[:1])
		self.generator_fool_model.train_on_batch( np.random.normal(0, 1, size=(1,LATENT)), y_train[:1] )
		prob2 = self.class_model.predict(X_train[:1])
		assert((prob1==prob2).all())
		self.class_model.trainable = True

def batch_to_jpeg(batch, labels, fn="test.png"):
	BATCH = batch.shape[0]
	batch_labels_best = []
	batch_labels_text = []
	for b in range(BATCH):
		best = labels[b].argmax()
		batch_labels_best.append( best )
		batch_labels_text.append( labels_text[best] )
	W = batch.shape[-1]
	H = batch.shape[-2]
	hor_n = 10
	ver_n = int( np.ceil(BATCH/hor_n) )
	BIG_W = W + 48
	BIG_H = H + 16
	pic = np.ones( shape=(3,ver_n*BIG_H,hor_n*BIG_W) )

	import PIL
	import PIL.Image as Image
	import PIL.ImageDraw as ImageDraw
	import PIL.ImageFont as ImageFont
	font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', 10)
	for y in range(ver_n):
		for x in range(hor_n):
			i = y*hor_n + x
			if i >= BATCH: break
			pic[:, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i]
	image = scipy.misc.toimage(pic, cmin=-1.0, cmax=1.0)
	draw = ImageDraw.Draw(image)
	for y in range(ver_n):
		for x in range(hor_n):
			i = y*hor_n + x
			if i >= BATCH: break
			draw.text((x*BIG_W, y*BIG_H+H), "%2.0f %s" % (labels[i,batch_labels_best[i]]*100, batch_labels_text[i]), font=font, fill='rgb(0, 0, 0)')
	image.save(fn)

def train(gan):
	cursor = 1e10
	epoch = 0
	while 1:
		if cursor+BATCH > X_train.shape[0]:
			sh = np.random.choice( X_train.shape[0], size=X_train.shape[0], replace=False )
			shuffled_x_train = X_train[sh]
			shuffled_y_train = y_train[sh]
			cursor = 0
			epoch += 1
			#loss, acc = gan.class_model.evaluate( X_test, y_test, batch_size=BATCH )
			#print "e%i test ---- loss=%0.3f acc=%0.2f%%" % (epoch, loss, acc*100)

			if epoch > 1:
				import shutil
				gan.generator_only_model.save_weights("_generator_weights.tmp")
				shutil.move("_generator_weights.tmp", "_generator_weights")
				gan.class_model.save_weights("_class_weights.tmp")
				shutil.move("_class_weights.tmp", "_class_weights")
				print("SNAPSHOT")

		x = shuffled_x_train[cursor:cursor+BATCH]
		y = shuffled_y_train[cursor:cursor+BATCH]
		cursor += BATCH

		x_merged = np.zeros( shape=(BATCH*2, 3, H, W) )
		x_merged[:BATCH] = x
		y_discrm = np.zeros( shape=(BATCH*2, LABELS+1) )
		y_discrm[:BATCH] = y
		y_genera = np.zeros( shape=(BATCH, LABELS+1) )
		latent = np.random.normal(0, 1, size=(BATCH,LATENT))
		for b in range(BATCH):
			i = latent[b,:LABELS].argmax()
			latent[b,:LABELS] = 0
			latent[b,i] = 1
			y_genera[b,i] = 1            # generator should try to improve classification towards i
			y_discrm[BATCH+b,LABELS] = 1 # discriminator should try to improve towards FAKE
		x_merged[BATCH:] = gan.generator_only_model.predict(latent)

		classifier_only = False
		if not classifier_only:
			x = x_merged
			y = y_discrm

		loss, acc = gan.class_model.train_on_batch(x, y)
		if cursor % 5000 == 0:
			print "e%i:%05i loss=%0.3f acc=%0.2f%%" % (epoch, cursor, loss, acc*100)
			labels = gan.class_model.predict( x )
			batch_to_jpeg(x_merged, y_discrm, "discriminator_task.png")
			batch_to_jpeg(x, labels, "classifier.png")

		if classifier_only: continue

		#labels1 = gan.class_model.predict( x_merged[BATCH:] )
		#labels2 = gan.generator_fool_model.predict( latent )
		#print "labels1", labels1[0]
		#print "labels2", labels2[0]

		test1 = gan.class_model.predict( x[:1] )
		loss, acc = gan.generator_fool_model.train_on_batch(latent, y_genera)
		test2 = gan.class_model.predict( x[:1] )
		#print test1[0]
		#print test2[0]
		#assert((test1==test2).all())

		if cursor % 5000 == 0:
			print "generator loss=%0.3f acc=%0.2f%%" % (loss, acc*100)
			labels = gan.generator_fool_model.predict( latent )
			batch_to_jpeg(x_merged[BATCH:], labels, "generator-real.png")

if __name__=="__main__":
	gan = Gan()
	#gan.class_model.summary()
	gan.generator_fool_model.summary()
	if sys.argv[1]=="zero":
		pass
	else:
		gan.generator_only_model.load_weights("_generator_weights")
		gan.class_model.load_weights("_class_weights")

	train(gan)

