import argparse, sys, time
import tensorflow as tf
import numpy as np

import datasets
dataset = datasets.Dataset()
dataset.BATCH = 100
datasets.mnist(dataset)
#cifar10()
print("HxW=%ix%i" % (dataset.W,dataset.H))

MAX_STEPS = 5000
LOG_DIR = "ramdisk/log"
LATENT = 150
LABELS = 10

# rescale to -0.95 .. +0.95
dataset.X_train *= 2.0
dataset.X_train -= 1.0
dataset.X_train *= 0.95
dataset.X_test *= 2.0
dataset.X_test -= 1.0
dataset.X_test *= 0.95

def glorot_normal(shape):
	s = np.sqrt(2. / (shape[-1] + shape[-2]))
	return tf.random_normal_initializer(stddev=s)

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def dense_layer(input_tensor, input_dim, output_dim, layer_name, act_fun=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = tf.get_variable("weights", [input_dim, output_dim], initializer=glorot_normal([input_dim, output_dim]))
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = tf.get_variable("bias_variable", [output_dim], initializer=tf.constant_initializer(0.0))
			variable_summaries(biases)
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			#tf.summary.histogram('pre_activations', preactivate)
			activations = act_fun(preactivate, name='activation')
			#tf.summary.histogram('activations', activations)
	return activations

def conv_layer(input, kernel_shape, bias_shape, stride=1, act_fun=tf.nn.relu):
	weights = tf.get_variable("weights", kernel_shape, initializer=glorot_normal(kernel_shape))
	biases  = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
	conv    = tf.nn.convolution(input, weights, strides=[stride,stride], padding='SAME')
	pre_act = conv + biases
	#pic_shape = act.get_shape()
	n_in  = kernel_shape[-2]
	n_out = kernel_shape[-1]

	#bn_mean     = tf.get_variable("bn_mean", batch_mean, initializer=tf.constant_initializer(0.0))
	#bn_variance = tf.get_variable("bn_variance", batch_var, initializer=tf.constant_initializer(0.5))
	#beta  = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',  trainable=False)
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=False)
	#ema = tf.train.ExponentialMovingAverage(decay=0.5)
	# https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412

	batch_mean, batch_var = tf.nn.moments(pre_act, [0,1,2], name='moments', keep_dims=True)
	#for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
	#for simple batch normalization pass axes=[0] (batch only).

	bn = tf.nn.batch_normalization(pre_act, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=0.01, name="bn")
	return act_fun(bn)

def discriminator_network(x, keep_prob):
	features = dataset.COLORS
	h,w = dataset.H,dataset.W

	with tf.variable_scope('conv1') as scope:
		conv1_relu = conv_layer(x, [3,3,features,32], [32])
		features = 32

	with tf.variable_scope('conv2_1') as scope:
		stride = 2
		conv2_1_relu = conv_layer(conv1_relu, [3,3,features,64], [64], stride=stride)
		h //= stride
		w //= stride
		features = 64
	with tf.variable_scope('conv2_2') as scope:
		conv2_2_relu = conv_layer(conv2_1_relu, [3,3,features,64], [64])

	with tf.variable_scope('conv3_1') as scope:
		stride = 2
		conv3_1_relu = conv_layer(conv2_2_relu, [3,3,features,128], [128], stride=stride)
		h //= stride
		w //= stride
		features = 128
	with tf.variable_scope('conv3_2') as scope:
		conv3_2_relu = conv_layer(conv3_1_relu, [3,3,features,128], [128])

	with tf.variable_scope('conv4_1') as scope:
		stride = 2
		conv4_1_relu = conv_layer(conv3_2_relu, [3,3,features,256], [256], stride=stride)
		h //= stride
		w //= stride
		features = 256
	print("final h w = %i %i" % (h,w))

	flat = tf.reshape(conv4_1_relu, [-1, h*w*features])

	with tf.name_scope('dropout'):
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(flat, keep_prob)

	with tf.variable_scope("dense1"):
		dense1 = dense_layer(dropped, h*w*features, 256, 'dense1')

	# without another dense 10-way layer, called is supposed to add one or more of these
	return dense1, 256

def do_all():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config=tf.ConfigProto(log_device_placement=True))
	sess = tf.InteractiveSession(config=config)

	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None,dataset.H,dataset.W,1], name='x-input')
		y_true = tf.placeholder(tf.float32, [None,LABELS], name='y-input')
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.image('input', x, 4)

	disc_wide_code, disc_width = discriminator_network(x, keep_prob)

	with tf.variable_scope("dense_to_classes"):
		y = dense_layer(disc_wide_code, disc_width, 10, 'dense2', act_fun=tf.identity)

	with tf.name_scope('cross_entropy'):
		diff = tf.nn.softmax_cross_entropy_with_logits(y, y_true)
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)
			tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
	test_writer = tf.summary.FileWriter(LOG_TEST_DIR)
	tf.global_variables_initializer().run()


	def feed_dict(train):
		if train:
			xs, ys = dataset.next_batch()
			k = 0.7
		else:
			xs, ys = dataset.X_test, dataset.y_test
			k = 1.0
		return { x: xs, y_true: ys, keep_prob: k }

	ts1 = 0
	for i in range(MAX_STEPS):
		if i % 100 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			ts2 = time.time()
			if ts1:
				print('e%02i:%05i %0.2fms acc=%0.2f' % (dataset.epoch, i, 1000*(ts2-ts1), acc*100))
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, loss = sess.run(
				[merged, train_step], feed_dict=feed_dict(True),
				options=run_options,
				run_metadata=run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary, i)
			ts1 = ts2
		else:
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, i)

	train_writer.close()
	test_writer.close()

def main(aaa):
	#with tf.device('/cpu:0'):
	do_all()

if __name__ == '__main__':
	import shutil, os
	LOG_TEST_DIR = LOG_DIR + "/test_%s" % sys.argv[1]
	LOG_TRAIN_DIR = LOG_DIR + "/train_%s" % sys.argv[1]
	shutil.rmtree(LOG_TEST_DIR, ignore_errors=True)
	shutil.rmtree(LOG_TRAIN_DIR, ignore_errors=True)
	os.makedirs(LOG_TEST_DIR)
	os.makedirs(LOG_TRAIN_DIR)

	#from tensorflow.python.client import device_lib
	#print( device_lib.list_local_devices() )
	unparsed = []
	tf.app.run(main=main) #, argv=[sys.argv[0]] + unparsed)

