import argparse, sys, time
import tensorflow as tf
import numpy as np
import rl_algs.common.tf_util as U
import rl_algs.common.tf_weightnorm as W

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
    return tf.maximum(0.2*x, x)

def deconv_wn(x, name, kernel_shape, num_filters, stride, wn_init):
    batch_size,h,w,features = x.get_shape().as_list()
    want_features = kernel_shape[-2]
    output_shape = [batch_size,stride*h,stride*w,want_features]
    #v = tf.get_variable(name+"/v", kernel_shape, tf.float32, tf.orthogonal_initializer(0.1))
    v = tf.get_variable(name+"/v", kernel_shape, tf.float32, initializer=tf.random_normal_initializer(0.01))
    g = tf.get_variable(name+"/g", [want_features], tf.float32, tf.constant_initializer(1.0))
    w = tf.reshape(g, [want_features,1]) * v / tf.stop_gradient(tf.sqrt(tf.reduce_sum( tf.square(v), axis=[0,1,2] )))
    b = tf.get_variable(name+"/b", [1,1,1,want_features], tf.float32, initializer=tf.constant_initializer(0.0))
    # kernel_shape [height, width, output_channels, in_channels]
    x = tf.add(
        tf.nn.conv2d_transpose(x, w, output_shape, strides=[1,stride,stride,1]),
        b, name=name )
    print(name, want_features, output_shape, x.get_shape().as_list())
    if wn_init is not None:
        wn_init.tune_magnitude_of_this.append(x)
        wn_init.by_mashing_that.append(g)
        wn_init.bias_for_centering.append(b)
        wn_init.all_trainable_vars += [v, g, b]
    return x

def bn(x, name=None):
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
    #for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
    #for simple batch normalization pass axes=[0] (batch only).
    x = tf.nn.batch_normalization(x, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=0.0001, name=name)
    return x

def discriminator_network(x, wn_init):
    h, w, features = x.get_shape().as_list()[-3:]
    print("discriminator input h w f = %i %i %i" % (h,w,features))
    x = U.conv2d(x,  32, "conv11", [3,3], [1,1])
    x = leaky_relu(x)
    x = U.conv2d(x,  64, "conv21", [3,3], [2,2])
    x = leaky_relu(bn(x))
    x = U.conv2d(x, 128, "conv31", [3,3], [2,2])
    x = leaky_relu(bn(x))
    x = U.conv2d(x, 256, "conv41", [3,3], [2,2])
    x = leaky_relu(bn(x))
    batch, h, w, features = x.get_shape().as_list()
    print("discriminator final b h w f = %s %i %i %i -> flat %i" % (batch,h,w,features, h*w*features))
    x = tf.reshape(x, [-1,h*w*features])
    x = leaky_relu( U.dense(x, 256, "dense1") )
    return x

def generator_network(l, wn_init):
    batch_size,features = l.get_shape().as_list()
    print("generator input b f = %s %i" % (batch_size,features))
    h,w,features = 4,4,256
    x = tf.nn.relu( W.dense_wn(l, 2*h*w*features, "latent_fc1", wn_init=wn_init) )
    x = tf.nn.relu( W.dense_wn(l,   h*w*features, "latent_fc2", wn_init=wn_init) )
    x = tf.reshape(x, [-1,h,w,features])
    want_features = 256
    x = tf.nn.relu( deconv_wn(x, "deconv1", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    features, want_features = want_features, 256
    x = tf.nn.relu( deconv_wn(x, "deconv2", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    features, want_features = want_features, 128
    x = tf.nn.relu( deconv_wn(x, "deconv3", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    x = tf.nn.tanh( W.conv2d_wn(x, dataset.COLORS, "tocolors",  [1,1], [1,1], wn_init=wn_init) )
    h, w, features = x.get_shape().as_list()[-3:]
    print("generator final h w f = %i %i %i" % (h,w,features))
    return x

def do_all():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.allow_soft_placement = True
    #config.gpu_options.allow_growth = True
    #config=tf.ConfigProto(log_device_placement=True))
    sess = tf.InteractiveSession(config=config)

    real = tf.placeholder(tf.float32, [None,dataset.H,dataset.W,1], name='real_input')

    classification_true = tf.placeholder(tf.float32, [None,dataset.LABELS], name='classification_true')
    klass_true = tf.placeholder(tf.float32, [None,dataset.LABELS], name='klass_true')
    real_or_fake_true   = tf.placeholder(tf.float32, [None,2], name='real_or_fake_true')
    latent_placeholder  = tf.placeholder(tf.float32, [dataset.BATCH,LATENT], name='latent')

    gen_wm_init = W.WeightNormInitializer()
    with tf.variable_scope("generator_network"):
        fake = generator_network(latent_placeholder, gen_wm_init)
        generator_learnable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        print("generator_learnable: \n %s" % "\n ".join(v.name for v in generator_learnable))

    real_concat_fake = U.concatenate([real,fake], axis=0)

    disc_wm_init = W.WeightNormInitializer()
    class_wm_init = W.WeightNormInitializer()
    with tf.variable_scope("discriminator_network"):
        last1 = discriminator_network(real, class_wm_init)
        x = U.dense(last1, dataset.LABELS, "dense10", weight_init=U.normc_initializer(1.0))
        klass_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=klass_true))
        klass_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(klass_true, 1)), tf.float32))

        tf.get_variable_scope().reuse_variables()
        last2  = discriminator_network(real_concat_fake, disc_wm_init)
        x = U.dense(last2, dataset.LABELS, "dense10")
        classification_logits   = tf.nn.softmax(x)
        classification_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=classification_true))
        classification_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(classification_true, 1)), tf.float32))

        discriminator_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        print("disc trainable: \n %s" % "\n ".join(v.name for v in discriminator_trainable))

    with tf.variable_scope("disc_losses"):
        x = U.dense(last2, 2, "dense_real_or_fake")
        real_or_fake_logits   = tf.nn.softmax(x)
        real_or_fake_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=real_or_fake_true))
        real_or_fake_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(real_or_fake_true, 1)), tf.float32))

    tf.summary.scalar('classification_loss', classification_loss)
    tf.summary.scalar('classification_accuracy', classification_accuracy)
    tf.summary.scalar('klass_loss', klass_loss)
    tf.summary.scalar('klass_accuracy', klass_accuracy)
    tf.summary.scalar('real_or_fake_loss', real_or_fake_loss)
    tf.summary.scalar('real_or_fake_accuracy', real_or_fake_accuracy)

    # summary
    gen_wm_init.dump_to_tf_summary()
    disc_wm_init.dump_to_tf_summary()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
    test_writer = tf.summary.FileWriter(LOG_TEST_DIR)

    # adams
    with tf.name_scope('adam_discriminator'):
        adam_discriminator = tf.train.AdamOptimizer(0.00001, beta1=0.5).minimize(
            klass_loss + real_or_fake_loss,
            var_list=discriminator_trainable)
    with tf.name_scope('adam_generator'):
        adam_generator = tf.train.AdamOptimizer(0.00001, beta1=0.5).minimize(
            (-1.0)*real_or_fake_loss + classification_loss,
            var_list=generator_learnable)
    assert( not (set(generator_learnable) & set(discriminator_trainable)) )

    tf.global_variables_initializer().run()
    # stop_gradient(2*x) - x

    ts1 = 0
    first_batch = True
    stable_feed_dict = None
    for step in range(MAX_STEPS):
        if step % 100 == 0:
            #summary, acc = sess.run([merged, classification_accuracy], feed_dict={ real: dataset.X_test, classification_true: dataset.y_test })
            #test_writer.add_summary(summary, step)
            acc = 1.0
            ts2 = time.time()
            if ts1:
                print('e%02i:%05i %0.2fms acc=%0.2f' % (dataset.epoch, step, 1000*(ts2-ts1), acc*100))
            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_options = None
            run_metadata = None   #tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        d_real, d_labels = dataset.next_batch()
        d_real  = d_real.copy()
        #d_real[:,:,:,:] += np.random.normal(0, 0.5, size=(dataset.BATCH,dataset.H,dataset.W,1))
        label_leak = 0.0
        d_latent         = np.random.normal(0, 1, size=(dataset.BATCH,LATENT) )
        d_class          = np.zeros( (2*dataset.BATCH,dataset.LABELS), dtype=np.float32 )
        d_class[:dataset.BATCH] = d_labels*(1-label_leak-label_leak/dataset.LABELS)
        d_class[:,:] += label_leak/dataset.LABELS

        real_or_fake     = np.zeros( (2*dataset.BATCH,2), dtype=np.float32 )
        real_or_fake[:dataset.BATCH,0] = 0.9
        real_or_fake[:dataset.BATCH,1] = 0.1
        real_or_fake[dataset.BATCH:,0] = 0
        real_or_fake[dataset.BATCH:,1] = 1.0

        for b in range(dataset.BATCH):
            i = d_latent[b,:dataset.LABELS].argmax()
            d_latent[b,:dataset.LABELS] = 0
            d_latent[b,i] = 1
            d_class[b+dataset.BATCH,i] = 1-label_leak

        feed_dict = {
            real: d_real,
            latent_placeholder: d_latent,
            klass_true: d_class[:dataset.BATCH],
            classification_true: d_class,
            real_or_fake_true: real_or_fake,
            }

        if first_batch:
            first_batch = False
            #gen_wm_init.data_based_initialization(feed_dict)
            #disc_wm_init.data_based_initialization({ real: dataset.X_test })
            class_wm_init.data_based_initialization({ real: dataset.X_test })
            stable_feed_dict = feed_dict

        if step % 100 == 0:
            summary, _, _ = sess.run(
                [merged, adam_discriminator, adam_generator],
                options=run_options,
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            train_writer.add_summary(summary, step)
            #if run_metadata:
            #    train_writer.add_run_metadata(run_metadata, 'step%05i' % step)
        else:
            _, _ = sess.run(
                [adam_discriminator, adam_generator],
                options=run_options,
                feed_dict=feed_dict)

        ts1 = ts2
        sys.stdout.write(".")
        sys.stdout.flush()

        if step % 10 == 0:
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

        if 0:
            real_concat_fake_export, classification_export = sess.run(
                [real_concat_fake, classification_logits],
                feed_dict = stable_feed_dict)
            datasets.batch_to_jpeg(dataset, real_concat_fake_export, classification_export, "movie/f%06i.png" % step)

    train_writer.close()
    test_writer.close()

def main(args):
    with tf.device('/gpu:1'):
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

