import tensorflow as tf
import numpy as np

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
	labels_text = "0 1 2 3 4 5 6 7 8 9 FAKE".split()
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

def cirar10(dataset):
	labels_text = "airplane automobile bird cat deer dog frog horse ship truck FAKE".split()
	dataset.COLORS = 3
	import keras.datasets.cifar10
	(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

def celebA(dataset):
	dataset.COLORS = 3
	dataset.W = 64
	dataset.H = 64

