import os, sys, shutil, time
if 0:
	os.environ["KERAS_BACKEND"] = "tensorflow"
	import tensorflow
else:
	os.environ['THEANO_FLAGS'] = "device=gpu,lib.cnmem=0.2"
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
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Merge, merge, Input
from keras.models import Sequential, Model
from keras.engine import Layer
from keras.optimizers import SGD, Adagrad, Adam, Adamax, RMSprop
from keras.constraints import nonneg
import keras.metrics
import scipy
import scipy.misc
import scipy.misc.pilutil

BATCH  = 50
LATENT = 150
H,W = 32,32
print("HxW=%ix%i" % (W,H))
LABELS = 10

if 0:
	import keras.datasets.cifar10
	(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
	labels_text = "airplane automobile bird cat deer dog frog horse ship truck FAKE".split()
	COLORS = 3
else:
	import keras.datasets.mnist
	(X_train_, y_train), (X_test_, y_test) = keras.datasets.mnist.load_data()
	labels_text = "0 1 2 3 4 5 6 7 8 9 FAKE".split()
	X_train = np.zeros( (len(X_train_),1,32,32) )
	X_test  = np.zeros( (len(X_test_),1,32,32) )
	X_train[:,0,2:30,2:30] = X_train_
	X_test[:,0,2:30,2:30] = X_test_
	COLORS = 1

print(X_train.shape)  # (50000, 3, 32, 32)
print(y_train)        # (50000, 1)            [[0], [1], [9]]  

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 128
X_train -= 1.0
X_train *= 0.9
X_test  /= 128
X_test  -= 1.0
X_test  *= 0.9

#X_train = X_train.reshape( (len(X_train),COLORS,H,W) )
y_train = y_train.reshape( (len(y_train),1) )
#X_test = X_test.reshape( (len(X_test),COLORS,H,W) )
y_test = y_test.reshape( (len(y_test),1) )

y_train = np_utils.to_categorical(y_train, LABELS+1).astype('float32')
y_test  = np_utils.to_categorical(y_test,  LABELS+1).astype('float32')

class Gan:
	def __init__(self):
		self.conv = Sequential()
		self.conv.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', batch_input_shape=(None,COLORS,H,W)))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2,2), border_mode='same'))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Convolution2D(128, 3, 3, activation='relu', subsample=(2,2), border_mode='same'))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Convolution2D(256, 3, 3, activation='relu', subsample=(2,2), border_mode='same'))
		self.conv.add(BatchNormalization(mode=0, axis=1))
		self.conv.add(Flatten())

		self.real_or_fake_model = Sequential()
		self.real_or_fake_model.add(self.conv)
		self.real_or_fake_model.add(Dense(256, activation='relu'))
		self.real_or_fake_model.add(Dense(2, activation='softmax'))

		self.additional_model.add(self.conv)
		self.additional_model.add(Dense(256, activation='relu'))
		self.additional_model.add(Dense(LABELS+1, activation='softmax'))

		self.additional_model = Sequential()

		#def modified_categorical_crossentropy(y_true, y_pred):
		#	_EPSILON = K.epsilon()
		#	return K.mean(K.sum(-y_true*K.log(K.clip(
		#		0.1 + y_pred*0.9/K.sum(y_pred, axis=-1, keepdims=True),
		#		_EPSILON, 1-_EPSILON)), axis=-1))


		self.class_model.compile(
			optimizer=Adam(lr=0.0002, beta_2=0.999),
			loss="categorical_crossentropy",
			#loss=modified_categorical_crossentropy,
			metrics=['categorical_accuracy'])
		self.class_model.train_on_batch( X_train[:1], y_train[:1] )  # this "finishes" the compilation
		for x in self.class_model.layers:
			x.trainable = False   # should do it before compiling generator

		alpha = 0.1
		self.generator_only_model = Sequential()
		self.generator_only_model.add(Dense(1024, batch_input_shape=(None,LATENT)))
		self.generator_only_model.add(LeakyReLU(alpha))
		self.generator_only_model.add(Dense(1024*W//8*H//8))
		self.generator_only_model.add(LeakyReLU(alpha))
		self.generator_only_model.add(Reshape( (1024,H//8,W//8) ))
		self.generator_only_model.add(UpSampling2D( size=(2,2) ))
		self.generator_only_model.add(Convolution2D(128, 3,3, border_mode='same', init="glorot_normal"))
		self.generator_only_model.add(LeakyReLU(alpha))
		#self.generator_only_model.add(BatchNormalization(mode=0, axis=1))
		self.generator_only_model.add(UpSampling2D( size=(2,2) ))
		self.generator_only_model.add(Convolution2D(64, 3,3, border_mode='same', init="glorot_normal"))
		self.generator_only_model.add(LeakyReLU(alpha))
		#self.generator_only_model.add(BatchNormalization(mode=0, axis=1))
		self.generator_only_model.add(UpSampling2D( size=(2,2) ))
		self.generator_only_model.add(Convolution2D(32, 3,3, border_mode='same', init="glorot_normal"))
		self.generator_only_model.add(LeakyReLU(alpha))
		#self.generator_only_model.add(BatchNormalization(mode=0, axis=1))
		self.generator_only_model.add(Convolution2D(COLORS, 3,3, activation='tanh', border_mode='same', init="glorot_normal"))

		self.class_model.trainable = False
		latent = Input( batch_shape=(None,LATENT), name="latent" )
		g = self.generator_only_model(latent)
		class_generator = self.class_model(g)

		self.generator_fool_model = Model( input=latent, output=[real_or_fake, additional] )
		self.generator_fool_model.compile(
			optimizer=Adam(lr=0.00002, beta_2=0.999),
			loss=["binary_crossentropy", "categorical_crossentropy"],
			#loss=modified_categorical_crossentropy
			)
		prob1 = self.class_model.predict(X_train[:1])
		self.generator_fool_model.train_on_batch( np.random.normal(0, 1, size=(1,LATENT)), y_train[:1] )
		prob2 = self.class_model.predict(X_train[:1])

		# Ensure we don't learn discriminator
		print(prob1)
		print(prob2)
		#assert((prob1==prob2).all())
		self.class_model.trainable = True

def batch_to_jpeg(batch, labels, fn="ramdisk/test.png"):
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
			if COLORS==1:
				pic[0, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i]
				pic[1, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i]
				pic[2, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i]
			else:
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
	same_x = X_train[:BATCH]
	same_latent = None
	x_merged = np.zeros( shape=(BATCH*2, COLORS, H, W) )
	y_discrm = np.zeros( shape=(BATCH*2, LABELS+1) )
	y_genera = np.zeros( shape=(BATCH, LABELS+1) )
	while 1:
		if cursor+BATCH > X_train.shape[0]:
			sh = np.random.choice( X_train.shape[0], size=X_train.shape[0], replace=False )
			shuffled_x_train = X_train[sh]
			shuffled_y_train = y_train[sh]
			cursor = 0
			epoch += 1

			loss, acc = gan.class_model.evaluate( X_test, y_test, batch_size=BATCH )
			print("e%i test ---- loss=%0.3f acc=%0.2f%%" % (epoch, loss, acc*100))

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

		ts1 = time.time()
		x_merged[:BATCH] = x
		y_discrm[:] = np.random.uniform( low=0, high=0.2, size=(2*BATCH, LABELS+1) )
		y_discrm[:BATCH] += 0.8*y
		#y_discrm[BATCH:] = 0.1/(LABELS+1)
		y_genera[:] = np.random.uniform( low=0, high=0.2, size=(BATCH, LABELS+1) )
		latent = np.random.normal(0, 1, size=(BATCH,LATENT))
		#latent = np.random.uniform(low=-1, high=1, size=(BATCH,LATENT))
		for b in range(BATCH):
			i = latent[b,:LABELS].argmax()
			latent[b,:LABELS] = 0
			latent[b,i] = 1
			y_genera[b,i] += 0.9             # generator should try to improve classification towards i
			y_discrm[BATCH+b,LABELS] += 0.9  # discriminator should try to improve towards FAKE
		if same_latent is None:
			same_latent = latent[:BATCH]

		x_merged[BATCH:] = gan.generator_only_model.predict(latent)  ###
		ts2 = time.time()

		#batch_to_jpeg(x_merged, y_discrm, "ramdisk/_discr-labels.png")
		#batch_to_jpeg(x_merged[BATCH:], y_genera, "ramdisk/_genera-labels.png")

		classifier_only = False
		if not classifier_only:
			x = x_merged
			y = y_discrm

		loss, acc = gan.class_model.train_on_batch(x, y)  ###
		ts3 = time.time()

		if cursor % 2000 == 0:
			print("e%i:%05i loss=%0.3f acc=%0.2f%%" % (epoch, cursor, loss, acc*100))
			labels = gan.class_model.predict( x )
			batch_to_jpeg(x, labels, "ramdisk/classifier.png")

		if classifier_only: continue

		gan.class_model.trainable = False
		#print(y_genera[:1])
		#print(labels1[:1])
		loss = gan.generator_fool_model.train_on_batch(latent, y_genera)  ###
		labels1 = gan.generator_fool_model.predict( latent )
		loss = 0.0
		acc = 0
		fake = 0
		for b in range(BATCH):
			ans = labels1[b].argmax()
			need = y_genera[b].argmax()
			if ans==need: acc += 1
			if ans==LABELS: fake += 1
			loss += -np.log(labels1[b,need]) * (1.0/BATCH)
		print("ans=%i need=%i acc=%i fake=%i loss=%0.2f" % (ans, need, acc, fake, loss))

		gan.class_model.trainable = True
		ts4 = time.time()

		#labels2 = gan.generator_fool_model.predict( latent[:1] )
		#correct = latent[0,:LABELS].argmax()
		#print("improve towards %i = %s" % (correct, labels_text[correct]))
		#print("target", y_genera[0])
		#print("labels1", labels1[0])
		#print("labels1[%i]=%0.4f" % (correct, labels1[0,correct]))
		#print("labels2", labels2[0])
		#print("labels2[%i]=%0.4f" % (correct, labels2[0,correct]))

		if cursor % 2000 == 0:
			print("generator loss=%0.3f acc=%0.2f%%" % (loss, acc*100))
			print("gen %0.0fms + disc %0.0fms + learn gen %0.0fms = %0.0fms" % (1000*(ts2-ts1), 1000*(ts3-ts2), 1000*(ts4-ts3), 1000*(ts4-ts1)))
			labels = gan.generator_fool_model.predict( latent )
			batch_to_jpeg(x_merged[BATCH:], labels, "ramdisk/generator-real.png")

		#same_x[:] = gan.generator_only_model.predict(same_latent)
		#labels1 = gan.generator_fool_model.predict(same_latent)
		#labels2 = gan.class_model.predict(same_x)
		#batch_to_jpeg(x_merged[BATCH:], labels1, "ramdisk/generator-always-the-same-labels1.png")
		#batch_to_jpeg(x_merged[BATCH:], labels2, "ramdisk/generator-always-the-same-labels2.png")

if __name__=="__main__":
	#with tensorflow.device('/gpu:1'):
	gan = Gan()
	#gan.class_model.summary()
	gan.generator_fool_model.summary()
	if sys.argv[1]=="zero":
		pass
	else:
		gan.generator_only_model.load_weights("_generator_weights")
		gan.class_model.load_weights("_class_weights")

	train(gan)

