import tensorflow as tf
import numpy as np

def rescale_minus_095_plus_095(dataset):
	dataset.X_train *= 2.0
	dataset.X_train -= 1.0
	dataset.X_train *= 0.95
	dataset.X_test *= 2.0
	dataset.X_test -= 1.0
	dataset.X_test *= 0.95

class Dataset:
	def __init__(self):
		self.cursor = 1e10
		self.epoch = 0
	def next_batch(self):
		if self.cursor+self.BATCH > self.X_train.shape[0]:
			sh = np.random.choice( self.X_train.shape[0], size=self.X_train.shape[0], replace=False )
			self.shuffled_x_train = self.X_train[sh]
			self.shuffled_y_train = self.y_train[sh]
			self.cursor = 0
			self.epoch += 1
		x = self.shuffled_x_train[self.cursor:self.cursor+self.BATCH]
		y = self.shuffled_y_train[self.cursor:self.cursor+self.BATCH]
		self.cursor += self.BATCH
		return x, y

def mnist(dataset):
	dataset.labels_text = "0 1 2 3 4 5 6 7 8 9 FAKE".split()
	dataset.LABELS = 10
	dataset.H = 32
	dataset.W = 32
	dataset.COLORS = 1
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("ramdisk/data", one_hot=True)
	# train
	dataset.X_train = np.zeros( (len(mnist.train.images),dataset.H,dataset.W,1) )
	dataset.X_train[:,2:30,2:30,:] = mnist.train.images.reshape( (-1,28,28,1) )
	dataset.y_train = mnist.train.labels
	# test
	dataset.X_test  = np.zeros( (len(mnist.test.images),dataset.H,dataset.W,1) )
	dataset.X_test[:,2:30,2:30,:] = mnist.test.images.reshape( (-1,28,28,1) )
	dataset.y_test  = mnist.test.labels
	rescale_minus_095_plus_095(dataset)

def cirar10(dataset):
	dataset.labels_text = "airplane automobile bird cat deer dog frog horse ship truck FAKE".split()
	dataset.COLORS = 3
	import keras.datasets.cifar10
	(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

def celebA(dataset):
	dataset.COLORS = 3
	dataset.W = 64
	dataset.H = 64


###################

def batch_to_jpeg(dataset, batch, labels, fn="ramdisk/test.png"):
	batch_labels_best = []
	batch_labels_text = []
	BATCH = len(batch)
	for b in range(BATCH):
		best = labels[b].argmax()
		batch_labels_best.append( best )
		batch_labels_text.append( dataset.labels_text[best] )
	W = batch.shape[-2]
	H = batch.shape[-3]
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
			if dataset.COLORS==1:
				pic[0, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,0]
				pic[1, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,0]
				pic[2, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,0]
			else:
				pic[0, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,0]
				pic[1, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,1]
				pic[2, y*BIG_H:y*BIG_H+H, x*BIG_W:x*BIG_W+W] = batch[i,:,:,2]
	import scipy
	import scipy.misc
	import scipy.misc.pilutil
	image = scipy.misc.toimage(pic, cmin=-1.0, cmax=1.0)
	draw = ImageDraw.Draw(image)
	for y in range(ver_n):
		for x in range(hor_n):
			i = y*hor_n + x
			if i >= BATCH: break
			draw.text((x*BIG_W, y*BIG_H+H), "%2.0f %s" % (labels[i,batch_labels_best[i]]*100, batch_labels_text[i]), font=font, fill='rgb(0, 0, 0)')
	image.save(fn)


