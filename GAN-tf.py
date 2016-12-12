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

	#with tf.name_scope('dropout'):
	#	tf.summary.scalar('dropout_keep_probability', keep_prob)
	#	dropped = tf.nn.dropout(flat, keep_prob)

	with tf.variable_scope("dense1"):
		dense1 = dense_layer(flat, h*w*features, 256, 'dense1')

	# without another dense 10-way layer, called is supposed to add one or more of these
	return dense1, 256

def leaky_relu(x):
	return tf.maximum(0.1*x, x)

def deconvolution_layer(x, kernel_shape, bias_shape, stride, output_shape, act_fun=tf.nn.relu):
	weights = tf.get_variable("weights", kernel_shape, initializer=glorot_normal(kernel_shape))
	biases  = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
	conv    = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1,stride,stride,1], name='conv2d_transpose')

#tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format='NHWC', name=None)

        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
	pre_act = conv + biases
	batch_mean, batch_var = tf.nn.moments(pre_act, [0,1,2], name='moments', keep_dims=True)
	bn = tf.nn.batch_normalization(pre_act, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=0.01, name="bn")
	return act_fun(bn)

def generator_network(l):
	with tf.variable_scope("proj"):
		proj = dense_layer(l, LATENT, 1024*4*4, 'latent_project')
	with tf.variable_scope("unflat"):
		not_flat = tf.reshape(proj, [-1, 4, 4, 1024])
	features = 1024
	h,w = 4,4

	want_features = 512
	with tf.variable_scope("deconv1"):
		batch_size = tf.shape(l)[0]
		dconv1 = deconvolution_layer(not_flat, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features])
	w,h = 8,8
	features = want_features

	want_features = 256
	with tf.variable_scope("deconv2"):
		batch_size = tf.shape(l)[0]
		dconv2 = deconvolution_layer(dconv1, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features])
	w,h = 16,16
	features = want_features

	want_features = 128
	with tf.variable_scope("deconv3"):
		batch_size = tf.shape(l)[0]
		dconv3 = deconvolution_layer(dconv2, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features])
	features = want_features

	want_features = dataset.COLORS
	with tf.variable_scope('deconv4') as scope:
		dconv4 = conv_layer(dconv3, [3,3,features,want_features], [want_features])

	return dconv4

def do_all():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config=tf.ConfigProto(log_device_placement=True))
	sess = tf.InteractiveSession(config=config)

	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None,dataset.H,dataset.W,1], name='x-input')
		y_true = tf.placeholder(tf.float32, [None,LABELS], name='y-input')
		keep_prob = tf.placeholder(tf.float32)
		latent = tf.placeholder(tf.float32, [None,LATENT], name='latent')
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
	generated = generator_network(latent)

	tf.global_variables_initializer().run()

	#tf.summary.image('generated', generated, 4)

	ts1 = 0
	for i in range(MAX_STEPS):
		if i % 100 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict={ x: dataset.X_test, y_true: dataset.y_test, keep_prob: 1.0 })
			test_writer.add_summary(summary, i)
			ts2 = time.time()
			if ts1:
				print('e%02i:%05i %0.2fms acc=%0.2f' % (dataset.epoch, i, 1000*(ts2-ts1), acc*100))
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
		else:
			run_options = None
			run_metadata = None
		xs, ys = dataset.next_batch()
		summary, _ = sess.run(
			[merged, train_step], feed_dict={ x: xs, y_true: ys, keep_prob: 0.7 },
			options=run_options,
			run_metadata=run_metadata)
		if run_metadata: train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
		train_writer.add_summary(summary, i)
		ts1 = ts2
		#summary, _ = sess.run([merged, train_step], feed_dict=feed_classifier(True))
		#train_writer.add_summary(summary, i)

		l = np.random.normal(0, 1, size=(dataset.BATCH,LATENT) )
		summary2 = tf.summary.image("generated", generated, 1)
		s2,gen_images = sess.run( [summary2,generated], feed_dict={ latent: l } )
		#summary2 = tf.Summary()
		#print(type(tf.summary))
		#print(dir(summary))
		#print(dir(summary2))
		#summary2.image("generated", gen_images, 1)
		train_writer.add_summary(s2)

	train_writer.close()
	test_writer.close()

def main(args):
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

