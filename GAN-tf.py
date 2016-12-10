import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("ramdisk/data", one_hot=True)

MAX_STEPS = 1000
LOG_DIR = "ramdisk/log"

def train():
	sess = tf.InteractiveSession()

	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_true = tf.placeholder(tf.float32, [None, 10], name='y-input')

	def feed_dict(train):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		if train:
			xs, ys = mnist.train.next_batch(100)
			k = 0.9
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return { x: xs, y_true: ys, keep_prob: k }

	with tf.name_scope('input_reshape'):
		image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image('input', image_shaped_input, 10)

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)
	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

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

	def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		with tf.name_scope(layer_name):
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, output_dim])
				variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = bias_variable([output_dim])
				variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('pre_activations', preactivate)
				activations = act(preactivate, name='activation')
				tf.summary.histogram('activations', activations)
		return activations

	hidden1 = dense_layer(x, 784, 500, 'layer1')

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(hidden1, 0.9)

	y = dense_layer(dropped, 500, 10, 'layer2', act=tf.identity)

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

	for i in range(MAX_STEPS):
		if i % 10 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		if i % 100 == 99:  # Record execution stats
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, loss = sess.run(
				[merged, train_step], feed_dict=feed_dict(True),
				options=run_options,
				run_metadata=run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary, i)
			print('Adding run metadata for', i)
		else:
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, i)

	train_writer.close()
	test_writer.close()

def main(aaa):
	train()

if __name__ == '__main__':
	import shutil, os
	LOG_TEST_DIR = LOG_DIR + "/test_%s" % sys.argv[1]
	LOG_TRAIN_DIR = LOG_DIR + "/train_%s" % sys.argv[1]
	shutil.rmtree(LOG_TEST_DIR, ignore_errors=True)
	shutil.rmtree(LOG_TRAIN_DIR, ignore_errors=True)
	os.makedirs(LOG_TEST_DIR)
	os.makedirs(LOG_TRAIN_DIR)
	unparsed = []
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

