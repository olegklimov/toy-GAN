import argparse, sys, time
import tensorflow as tf
import numpy as np

import datasets
dataset = datasets.Dataset()
dataset.BATCH = 80
datasets.mnist(dataset)
#cifar10()
print("HxW=%ix%i" % (dataset.W,dataset.H))

MAX_STEPS = 9000
LOG_DIR = "ramdisk/log"
LATENT = 150

def glorot_normal(shape):
    s = np.sqrt(2. / (shape[-1] + shape[-2]))
    return tf.random_normal_initializer(stddev=s)

def leaky_relu(x):
    return tf.maximum(0.1*x, x)

def variable_summaries(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    #with tf.name_scope('stddev'):
    #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #    tf.summary.scalar('stddev', stddev)
    #    tf.summary.scalar('max', tf.reduce_max(var))
    #    tf.summary.scalar('min', tf.reduce_min(var))
    #    tf.summary.histogram('histogram', var)

def dense_layer(learnable, input_tensor, input_dim, output_dim, act_fun=tf.nn.relu):
    with tf.name_scope('weights'):
        weights = tf.get_variable("weights", [input_dim, output_dim], initializer=glorot_normal([input_dim, output_dim]))
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases = tf.get_variable("bias_variable", [output_dim], initializer=tf.constant_initializer(0.0))
        variable_summaries(biases)
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act_fun(preactivate)
    #tf.summary.histogram('activations', activations)
    learnable.append(weights)
    learnable.append(biases)
    return activations

def conv_layer(learnable, input, kernel_shape, bias_shape, stride=1, act_fun=tf.nn.relu, want_bn=True):
    with tf.name_scope('weights'):
        weights = tf.get_variable("weights", kernel_shape, initializer=glorot_normal(kernel_shape))
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases  = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        variable_summaries(biases)
    conv    = tf.nn.convolution(input, weights, strides=[stride,stride], padding='SAME')
    pre_act = conv + biases

    if want_bn:
        #pic_shape = act.get_shape()
        n_in  = kernel_shape[-2]
        n_out = kernel_shape[-1]

        #bn_mean     = tf.get_variable("bn_mean", batch_mean, initializer=tf.constant_initializer(0.0))
        #bn_variance = tf.get_variable("bn_variance", batch_var, initializer=tf.constant_initializer(0.5))
        #beta  = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',  trainable=False)
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=False)
        #ema = tf.train.ExponentialMovingAverage(decay=0.5)
        # https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412

        batch_mean, batch_var = tf.nn.moments(pre_act, [0,1,2], name='moments') #, keep_dims=True)
        #for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
        #for simple batch normalization pass axes=[0] (batch only).

        bn = tf.nn.batch_normalization(pre_act, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=0.0001, name="bn")
        ret = act_fun(bn)
    else:
        ret = act_fun(pre_act)

    learnable.append(weights)
    learnable.append(biases)
    return ret

def deconvolution_layer(learnable, x, kernel_shape, bias_shape, stride, output_shape, act_fun=tf.nn.relu):
    with tf.name_scope('weights'):
        weights = tf.get_variable("weights", kernel_shape, initializer=glorot_normal(kernel_shape))
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases  = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        variable_summaries(biases)

    conv    = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1,stride,stride,1], name='conv2d_transpose')
    #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    pre_act = conv + biases
    batch_mean, batch_var = tf.nn.moments(pre_act, [0,1,2], name='moments', keep_dims=True)
    bn = tf.nn.batch_normalization(pre_act, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=0.01, name="bn")

    learnable.append(weights)
    learnable.append(biases)
    return act_fun(bn)

########################

def discriminator_network(learnable, x):
    features = dataset.COLORS
    h,w = dataset.H,dataset.W

    with tf.variable_scope('conv1') as scope:
        conv1_relu = conv_layer(learnable, x, [3,3,features,32], [32], act_fun=leaky_relu)
        features = 32

    with tf.variable_scope('conv2_1') as scope:
        stride = 2
        conv2_1_relu = conv_layer(learnable, conv1_relu, [3,3,features,64], [64], stride=stride, act_fun=leaky_relu)
        h //= stride
        w //= stride
        features = 64
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_relu = conv_layer(learnable, conv2_1_relu, [3,3,features,64], [64], act_fun=leaky_relu)

    with tf.variable_scope('conv3_1') as scope:
        stride = 2
        conv3_1_relu = conv_layer(learnable, conv2_2_relu, [3,3,features,128], [128], stride=stride, act_fun=leaky_relu)
        h //= stride
        w //= stride
        features = 128
    with tf.variable_scope('conv3_2') as scope:
        conv3_2_relu = conv_layer(learnable, conv3_1_relu, [3,3,features,128], [128], act_fun=leaky_relu)

    with tf.variable_scope('conv4_1') as scope:
        stride = 2
        conv4_1_relu = conv_layer(learnable, conv3_2_relu, [3,3,features,256], [256], stride=stride, act_fun=leaky_relu)
        h //= stride
        w //= stride
        features = 256
    print("discriminator final h w = %i %i" % (h,w))

    flat = tf.reshape(conv4_1_relu, [-1, h*w*features])

    with tf.variable_scope("dense1"):
        dense1 = dense_layer(learnable, flat, h*w*features, 256, act_fun=leaky_relu)

    # without another dense 10-way layer, called is supposed to add one or more of these
    return dense1, 256

def generator_network(learnable, l):
    act = tf.nn.relu
    with tf.variable_scope("proj"):
        proj = dense_layer(learnable, l, LATENT, 1024*4*4, act_fun=act)
    with tf.variable_scope("unflat"):
        not_flat = tf.reshape(proj, [-1, 4, 4, 1024])
    features = 1024
    h,w = 4,4

    want_features = 512
    with tf.variable_scope("deconv1"):
        batch_size = tf.shape(l)[0]
        dconv1 = deconvolution_layer(learnable, not_flat, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features], act_fun=act)
    w,h = 8,8
    features = want_features

    want_features = 256
    with tf.variable_scope("deconv2"):
        batch_size = tf.shape(l)[0]
        dconv2 = deconvolution_layer(learnable, dconv1, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features], act_fun=act)
    w,h = 16,16
    features = want_features

    want_features = 128
    with tf.variable_scope("deconv3"):
        batch_size = tf.shape(l)[0]
        dconv3 = deconvolution_layer(learnable, dconv2, [3,3,want_features,features], [want_features], 2, [batch_size,2*h,2*w,want_features], act_fun=act)
    features = want_features

    want_features = dataset.COLORS
    with tf.variable_scope('deconv4') as scope:
        dconv4 = conv_layer(learnable, dconv3, [3,3,features,want_features], [want_features], act_fun=tf.nn.tanh, want_bn=False)

    return dconv4  # *1.1 make data -1.1 .. +1.1 to reach 1 easier

def do_all():
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #config.gpu_options.allow_growth = True
    #config=tf.ConfigProto(log_device_placement=True))
    sess = tf.InteractiveSession(config=config)

    real = tf.placeholder(tf.float32, [None,dataset.H,dataset.W,1], name='real-input')

    classification_true = tf.placeholder(tf.float32, [None,dataset.LABELS], name='classification_true')
    real_or_fake_true = tf.placeholder(tf.float32, [None,2], name='real_or_fake_true')

    latent_placeholder = tf.placeholder(tf.float32, [None,LATENT], name='latent')

    generator_learnable = []
    with tf.name_scope("generator-network"):
        fake = generator_network(generator_learnable, latent_placeholder)

    real_concat_fake = tf.concat(0, [real,fake])

    discriminator_learnable = []
    with tf.name_scope("discriminator-network"):
        disc_wide_code, disc_width = discriminator_network(discriminator_learnable, real_concat_fake)

    # classes
    with tf.name_scope("classification"):
        with tf.variable_scope("dense_classification"):
            classification = dense_layer(discriminator_learnable, disc_wide_code, disc_width, dataset.LABELS, act_fun=tf.identity)
        classification_logits  = tf.nn.softmax(classification)
        classification_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classification, classification_true))
        classification_correct  = tf.equal(tf.argmax(classification, 1), tf.argmax(classification_true, 1))
        classification_accuracy = tf.reduce_mean(tf.cast(classification_correct, tf.float32))
        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('classification_accuracy', classification_accuracy)
        #tf.summary.histogram('classification_logits', classification_logits)

    # real or fake
    with tf.name_scope("real_or_fake"):
        with tf.variable_scope("dense_real_or_fake"):
            real_or_fake = dense_layer(discriminator_learnable, disc_wide_code, disc_width, 2, act_fun=tf.identity)
        real_or_fake_logits  = tf.nn.softmax(real_or_fake)
        real_or_fake_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(real_or_fake, real_or_fake_true))
        real_or_fake_correct  = tf.equal(tf.argmax(real_or_fake, 1), tf.argmax(real_or_fake_true, 1))
        real_or_fake_accuracy = tf.reduce_mean(tf.cast(real_or_fake_correct, tf.float32))
        tf.summary.scalar('real_or_fake_loss', real_or_fake_loss)
        tf.summary.scalar('real_or_fake_accuracy', real_or_fake_accuracy)
        #tf.summary.histogram('real_or_fake_logits', real_or_fake_logits)

    # summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
    test_writer = tf.summary.FileWriter(LOG_TEST_DIR)

    # adams
    with tf.name_scope('adam_discriminator'):
        adam_discriminator = tf.train.AdamOptimizer(0.00005, beta1=0.5).minimize(
            classification_loss + real_or_fake_loss,
            var_list=discriminator_learnable)
    with tf.name_scope('adam_generator'):
        adam_generator = tf.train.AdamOptimizer(0.00005, beta1=0.5).minimize(
            classification_loss + (-1.0)*real_or_fake_loss,
            var_list=generator_learnable)

    tf.global_variables_initializer().run()

# stop_gradient(2*x) - x

    ts1 = 0
    for step in range(MAX_STEPS):
        if step % 100 == 0:
            #summary, acc = sess.run([merged, classification_accuracy], feed_dict={ real: dataset.X_test, classification_true: dataset.y_test })
            #test_writer.add_summary(summary, step)
            acc = 1.0
            ts2 = time.time()
            if ts1:
                print('e%02i:%05i %0.2fms acc=%0.2f' % (dataset.epoch, step, 1000*(ts2-ts1), acc*100))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        d_real, d_labels = dataset.next_batch()
        d_real  = d_real.copy()
        d_real[:,:,:,:] += np.random.uniform(low=-1.0, high=+1.0, size=(dataset.BATCH,dataset.H,dataset.W,1))
        label_leak = 0.2
        d_latent         = np.random.normal(0, 1, size=(dataset.BATCH,LATENT) )
        d_class          = np.zeros( (2*dataset.BATCH,dataset.LABELS), dtype=np.float32 )
        d_class[:dataset.BATCH] = d_labels*(1-label_leak-label_leak/dataset.LABELS)
        d_class[:,:] += label_leak/dataset.LABELS

        real_or_fake     = np.zeros( (2*dataset.BATCH,2), dtype=np.float32 )
        real_or_fake[:dataset.BATCH,0] = 0.7
        real_or_fake[:dataset.BATCH,1] = 0.3
        real_or_fake[dataset.BATCH:,0] = 0.3
        real_or_fake[dataset.BATCH:,1] = 0.7

        for b in range(dataset.BATCH):
            i = d_latent[b,:dataset.LABELS].argmax()
            d_latent[b,:dataset.LABELS] = 0
            d_latent[b,i] = 1
            d_class[b+dataset.BATCH,i] = 1-label_leak

        summary, _, _ = sess.run(
            [merged, adam_discriminator, adam_generator],
            feed_dict = {
                real: d_real,
                latent_placeholder: d_latent,
                classification_true: d_class,
                real_or_fake_true: real_or_fake,
                },
            options=run_options,
            run_metadata=run_metadata)

        if run_metadata:
            train_writer.add_run_metadata(run_metadata, 'step%05i' % step)
        train_writer.add_summary(summary, step)
        ts1 = ts2
        sys.stdout.write(".")
        sys.stdout.flush()

        if step % 50 == 0:
            sys.stdout.write("j")
            sys.stdout.flush()
            real_concat_fake_export, classification_export = sess.run(
                [real_concat_fake, classification_logits],
                feed_dict = {
                    real: d_real,
                    latent_placeholder: d_latent,
                    classification_true: d_class,
                    real_or_fake_true: real_or_fake,
                    })
            sys.stdout.write("j")
            sys.stdout.flush()
            datasets.batch_to_jpeg(dataset, real_concat_fake_export, classification_export)

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
    tf.app.run(main=main)

