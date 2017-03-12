import argparse, sys, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
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

def p(v):
    print( v, v.get_shape().as_list() )

def glorot_normal(shape):
    s = np.sqrt(2. / (shape[-1] + shape[-2]))
    return tf.random_normal_initializer(stddev=s)

def leaky_relu(x):
    return tf.maximum(0.2*x, x)

def deconv_wn(x, name, kernel_shape, num_filters, stride, wn_init):
    batch_size,h,w,features = x.get_shape().as_list()
    want_features = kernel_shape[-2]
    output_shape = [batch_size,stride*h,stride*w,want_features]
    v = tf.get_variable(name+"/v", kernel_shape, tf.float32, initializer=tf.random_normal_initializer(0.01))
    g = tf.get_variable(name+"/g", [want_features], tf.float32, tf.constant_initializer(1.0))
    w = tf.reshape(g, [want_features,1]) * v / tf.stop_gradient(tf.sqrt(tf.reduce_sum( tf.square(v), axis=[0,1,2] )))
    b = tf.get_variable(name+"/b", [1,1,1,want_features], tf.float32, initializer=tf.constant_initializer(0.0))
    # kernel_shape [height, width, output_channels, in_channels]
    x = tf.add(
        tf.nn.conv2d_transpose(x, w, output_shape, strides=[1,stride,stride,1]),
        b, name=name )
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
    #print("discriminator input h w f = %i %i %i" % (h,w,features))
    x = U.conv2d(x,  32, "conv11", [3,3], [1,1])
    x = leaky_relu(x)
    x = U.conv2d(x,  64, "conv21", [3,3], [2,2])
    x = leaky_relu(bn(x))
    x = U.conv2d(x, 128, "conv31", [3,3], [2,2])
    x = leaky_relu(bn(x))
    x = U.conv2d(x, 256, "conv41", [3,3], [2,2])
    x = leaky_relu(bn(x))
    batch, h, w, features = x.get_shape().as_list()
    #print("discriminator final b h w f = %s %i %i %i -> flat %i" % (batch,h,w,features, h*w*features))
    x = tf.reshape(x, [-1,h*w*features])
    x = leaky_relu( U.dense(x, 256, "dense1") )
    return x

def generator_network(l, wn_init):
    batch_size,features = l.get_shape().as_list()
    #print("generator input b f = %s %i" % (batch_size,features))
    h,w,features = 4,4,256
    x = l
    #x = tf.nn.relu( W.dense_wn(l, 2*h*w*features, "latent_fc1", wn_init=wn_init) )
    x = tf.nn.relu( W.dense_wn(x,   h*w*features, "latent_fc2", wn_init=wn_init) )
    x = tf.reshape(x, [-1,h,w,features])
    want_features = 256
    x = tf.nn.relu( deconv_wn(x, "deconv1", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    features, want_features = want_features, 256
    x = tf.nn.relu( deconv_wn(x, "deconv2", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    features, want_features = want_features, 128
    x = tf.nn.relu( deconv_wn(x, "deconv3", [3,3,want_features,features], features, 2, wn_init=wn_init) )
    x = tf.nn.tanh( W.conv2d_wn(x, dataset.COLORS, "tocolors",  [1,1], [1,1], wn_init=wn_init) )
    h, w, features = x.get_shape().as_list()[-3:]
    #print("generator final h w f = %i %i %i" % (h,w,features))
    return x

class Tower:
    def __init__(self, reuse, BATCH_PER_GPU):
        self.real                = tf.placeholder(tf.float32, [BATCH_PER_GPU,dataset.H,dataset.W,1], name='real_input')
        self.klass_true          = tf.placeholder(tf.float32, [BATCH_PER_GPU,dataset.LABELS], name='klass_true')
        self.classification_true = tf.placeholder(tf.float32, [2*BATCH_PER_GPU,dataset.LABELS], name='classification_true')
        self.real_or_fake_true   = tf.placeholder(tf.float32, [2*BATCH_PER_GPU,2], name='real_or_fake_true')
        self.latent_placeholder  = tf.placeholder(tf.float32, [BATCH_PER_GPU,LATENT], name='latent')

        self.gen_wm_init = W.WeightNormInitializer()
        with tf.variable_scope("generator_network", reuse=reuse):
            fake = generator_network(self.latent_placeholder, self.gen_wm_init)
            self.generator_learnable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            #print("generator_learnable: \n %s" % "\n ".join( ["%s\t%s" % (v.name, str(v.get_shape().as_list())) for v in self.generator_learnable] ))

        self.real_concat_fake = U.concatenate([self.real,fake], axis=0)

        self.disc_wm_init = W.WeightNormInitializer()
        self.class_wm_init = W.WeightNormInitializer()
        with tf.variable_scope("discriminator_network", reuse=reuse):
            last1 = discriminator_network(self.real, self.class_wm_init)
            x = U.dense(last1, dataset.LABELS, "dense10", weight_init=U.normc_initializer(1.0))
            self.klass_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=self.klass_true))
            self.klass_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(self.klass_true, 1)), tf.float32))

            tf.get_variable_scope().reuse_variables()
            last2  = discriminator_network(self.real_concat_fake, self.disc_wm_init)
            x = U.dense(last2, dataset.LABELS, "dense10")
            self.classification_logits   = tf.nn.softmax(x)
            self.classification_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=self.classification_true))
            self.classification_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(self.classification_true, 1)), tf.float32))

            self.discriminator_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            #print("disc trainable: \n %s" % "\n ".join(v.name for v in self.discriminator_trainable))

        with tf.variable_scope("disc_losses", reuse=reuse):
            x = U.dense(last2, 2, "dense_real_or_fake")
            self.real_or_fake_logits   = tf.nn.softmax(x)
            self.real_or_fake_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=self.real_or_fake_true))
            self.real_or_fake_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(self.real_or_fake_true, 1)), tf.float32))

        assert( not (set(self.generator_learnable) & set(self.discriminator_trainable)) )

        self.loss_disc = self.klass_loss + self.real_or_fake_loss
        self.loss_gen  = self.classification_loss - self.real_or_fake_loss
        self.grad_disc = U.flatgrad(self.loss_disc, self.discriminator_trainable)
        self.grad_gen  = U.flatgrad(self.loss_gen, self.generator_learnable)

def train():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    #config.log_device_placement = True
    #config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=config)

    tower = []
    reuse = False
    GPUs = 1
    BATCH_PER_GPU = dataset.BATCH // GPUs
    assert BATCH_PER_GPU * GPUs == dataset.BATCH
    for d in range(GPUs):
        with tf.device("gpu:%i" % d):
            tower.append(Tower(reuse, BATCH_PER_GPU))
            reuse = True
            #flatgrad(loss, var_list):

    from rl_algs.common.mpi_adam import MpiAdam
    with tf.device("cpu:0"):
        with tf.name_scope('adam_discriminator'):
            adam_discriminator = MpiAdam(tower[0].discriminator_trainable, 0.00001, beta1=0.5)
        with tf.name_scope('adam_generator'):
            adam_generator = MpiAdam(tower[0].generator_learnable, 0.00001, beta1=0.5)

    tf.summary.scalar('classification_loss', tower[0].classification_loss)
    tf.summary.scalar('classification_accuracy', tower[0].classification_accuracy)
    tf.summary.scalar('klass_loss', tower[0].klass_loss)
    tf.summary.scalar('klass_accuracy', tower[0].klass_accuracy)
    tf.summary.scalar('real_or_fake_loss', tower[0].real_or_fake_loss)
    tf.summary.scalar('real_or_fake_accuracy', tower[0].real_or_fake_accuracy)

    # summary
    tower[0].gen_wm_init.dump_to_tf_summary()
    tower[0].disc_wm_init.dump_to_tf_summary()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
    test_writer = tf.summary.FileWriter(LOG_TEST_DIR)

    tf.global_variables_initializer().run()

    ts1 = 0
    first_batch = True
    stable_feed_dict = None
    for step in range(MAX_STEPS):
        if step % 100 == 0:
            #summary, acc = sess.run([merged, classification_accuracy], feed_dict={ real: dataset.X_test, self.classification_true: dataset.y_test })
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

        #feed_dict = {}
        #for d in range(GPUs):
        #    feed_dict[tower[d].real]                =       d_real[d*BATCH_PER_GPU:(d+1)*BATCH_PER_GPU]
        #    feed_dict[tower[d].latent_placeholder]  =     d_latent[d*BATCH_PER_GPU:(d+1)*BATCH_PER_GPU]
        #    feed_dict[tower[d].klass_true]          =      d_class[(2*d)*BATCH_PER_GPU:(2*d+1)*BATCH_PER_GPU]
        #    feed_dict[tower[d].classification_true] =      d_class[(2*d)*BATCH_PER_GPU:(2*d+2)*BATCH_PER_GPU]
        #    feed_dict[tower[d].real_or_fake_true]   = real_or_fake[(2*d)*BATCH_PER_GPU:(2*d+2)*BATCH_PER_GPU]
        feed_dict = {
            tower[0].real: d_real,
            tower[0].latent_placeholder: d_latent,
            tower[0].klass_true: d_class[:dataset.BATCH],
            tower[0].classification_true: d_class,
            tower[0].real_or_fake_true: real_or_fake,
            }

        if first_batch:
            first_batch = False
            #gen_wm_init.data_based_initialization(feed_dict)
            #disc_wm_init.data_based_initialization({ real: dataset.X_test })
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            tower[0].class_wm_init.data_based_initialization({ tower[0].real: dataset.X_test })
            stable_feed_dict = feed_dict

        todo = (
            [ tower[d].grad_gen  for d in range(GPUs) ] +
            [ tower[d].grad_disc for d in range(GPUs) ]
            # + [ tower[0].loss_disc, tower[0].loss_gen ]
            )

        if step % 100 == 0: todo.append(merged)
        t = sess.run( todo, feed_dict=feed_dict )
        if step % 100 == 0: train_writer.add_summary(t[-1], step)
        #loss_disc, loss_gen = t[-2:]

        print("todo", todo, "t", t)

        grad_disc = np.sum( t[GPUs:2*GPUs], axis=0 )
        grad_gen  = np.sum( t[:GPUs], axis=0 )
        adam_discriminator.update( grad_disc )
        adam_generator.update( grad_gen )


        ts1 = ts2
        sys.stdout.write(".")
        sys.stdout.flush()

        if step % 10 == 0:
            sys.stdout.write("j")
            sys.stdout.flush()
            real_concat_fake_export, classification_export = sess.run(
                [tower[0].real_concat_fake, tower[0].classification_logits],
                feed_dict = feed_dict
                #{
                #    real: d_real,
                #    tower[0].latent_placeholder: d_latent,
                #    tower[0].classification_true: d_class,
                #    tower[0].real_or_fake_true: real_or_fake,
                #    }
                )
            sys.stdout.write("j")
            sys.stdout.flush()
            datasets.batch_to_jpeg(dataset, real_concat_fake_export, classification_export)

        if 0:
            real_concat_fake_export, classification_export = sess.run(
                [real_concat_fake, tower[0].classification_logits],
                feed_dict = stable_feed_dict)
            datasets.batch_to_jpeg(dataset, real_concat_fake_export, classification_export, "movie/f%06i.png" % step)

    train_writer.close()
    test_writer.close()

def main(args):
    #with tf.device('/cpu:0'):
    train()

if __name__ == '__main__':
    import shutil, os
    LOG_TEST_DIR = LOG_DIR + "/test_%s" % sys.argv[1]
    LOG_TRAIN_DIR = LOG_DIR + "/train_%s" % sys.argv[1]
    shutil.rmtree(LOG_TEST_DIR, ignore_errors=True)
    shutil.rmtree(LOG_TRAIN_DIR, ignore_errors=True)
    os.makedirs(LOG_TEST_DIR)
    os.makedirs(LOG_TRAIN_DIR)
    tf.app.run(main=main)

