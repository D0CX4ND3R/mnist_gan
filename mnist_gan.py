import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib import slim


PROJECT_NAME = 'gan_mnist'
DATASET_PATH = '/workspace/datasets/mnist/mnist.npz'
LOG_DIR = '/workspace/logs'
MODEL_DIR = '/workspace/models'
NORM_PARAMS = {'decay': 0.995, 'epsilon': 0.0001}
L2_WEIGHTS = 0.0005
BATCH_SIZE = 256
IMAGE_SIZE = [28, 28]
LEARNING_RATE = 0.003
TOTAL_EPOCHES = 150000
DISPLAY_ROW = 16


def load_mnist(dataset_path):
    mnist = np.load(dataset_path)
    x_train = mnist['x_train']
    y_train = mnist['y_train']
    x_test = mnist['x_test']
    y_test = mnist['y_test']
    mnist.close()
    return x_train, y_train, x_test, y_test


def batch_generator(image, batch_size=128):
    n, h, w = image.shape
    while True:
        ind = np.random.choice(np.arange(0, n, dtype=np.int32), batch_size, replace=False)
        batch_image = image[ind].astype(np.float32).reshape([batch_size, h, w, 1])
        batch_image = (batch_image - 127.5) / 127.5
        batch_noise = np.random.uniform(-1, 1, batch_image.shape)
        yield batch_image, batch_noise


def gan_generator(noise_image):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params=NORM_PARAMS,
                            weights_regularizer=slim.l2_regularizer(L2_WEIGHTS)):

            # 28 x 28 => 14 x 14
            conv1 = slim.repeat(noise_image, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=8, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

            # 14 x 14 => 7 x 7
            conv2 = slim.repeat(pool1, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=16, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

            conv3 = slim.repeat(pool2, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=32, scope='conv3')

            up4 = upsample_and_concat(conv3, conv2, 16, 32)
            conv4 = slim.repeat(up4, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=16, scope='conv4')

            up5 = upsample_and_concat(conv4, conv1, 8, 16)
            conv5 = slim.repeat(up5, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=8, scope='conv5')

            generated_image = slim.conv2d(conv5, 1, [1, 1], activation_fn=tf.nn.tanh)
    return generated_image


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output


def gan_discriminator(image):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.leaky_relu,
                            weights_regularizer=slim.l2_regularizer(L2_WEIGHTS)):
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3],
                                normalizer_fn=slim.batch_norm, normalizer_params=NORM_PARAMS):
                # 28 x 28 => 14 x 14
                net = slim.repeat(image, 2, slim.conv2d, num_outputs=16, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                # 14 x 14 => 7 x 7
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=32, scope='conv2')

                # global average pooling
                net = tf.reduce_mean(net, axis=[1, 2], name='global_average_pool')

                net = slim.flatten(net)
                net = slim.fully_connected(net, 64, scope='fc3')
                net = slim.fully_connected(net, 2, activation_fn=None, scope='fc4')
                logits = slim.softmax(net)
                pred = tf.argmax(slim.softmax(logits), axis=1)
    return logits, pred


def build_loss(logits_real, logits_fake):
    discriminator_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_real, labels=tf.ones(BATCH_SIZE, dtype=tf.int32)))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.zeros(BATCH_SIZE, dtype=tf.int32)))
    discriminate_loss = discriminator_loss_real + discriminator_loss_fake

    generator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.ones(BATCH_SIZE, dtype=tf.int32)))

    return discriminate_loss, generator_loss


def network(batch_images, batch_noises):
    logits_real, predict_real = gan_discriminator(batch_images)
    generate_images = gan_generator(batch_noises)
    logits_fake, predict_fake = gan_discriminator(generate_images)

    d_loss, g_loss = build_loss(logits_real, logits_fake)
    return d_loss, g_loss, generate_images


def display_batch_py(images, rows=8):
    n, h, w, c = images.shape
    images = images.reshape([rows, n // rows, h, w, c]).transpose(0, 2, 1, 3, 4).reshape([1, rows * h, n // rows * w, c])
    images = images * 127.5 + 127.5
    return images.astype(np.uint8)

def _main():
    with tf.name_scope('Inputs'):
        batch_images = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1], name='images')
        batch_noises = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 1], name='noises')

    d_loss, g_loss, g_images = network(batch_images, batch_noises)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope('Summary'):
        tf.summary.scalar('generator_loss', g_loss)
        tf.summary.scalar('discriminator_loss', d_loss)

        input_images = tf.py_func(display_batch_py, [batch_images], tf.uint8)
        generate_images = tf.py_func(display_batch_py, [g_images], tf.uint8)

        tf.summary.image('input_images', input_images)
        tf.summary.image('generate_images', generate_images)

    with tf.name_scope('Train'):
        global_step = tf.train.get_or_create_global_step()
        # d_train_op = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(d_loss, global_step)
        # g_train_op = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(g_loss, global_step)
        d_train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, global_step)
        g_train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, global_step)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()

    x_train, _, _, _ = load_mnist(DATASET_PATH)
    batch_gen = batch_generator(x_train, batch_size=BATCH_SIZE)
    saver = tf.train.Saver(max_to_keep=4)

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    log_path = os.path.join(LOG_DIR, '{}_{}'.format(PROJECT_NAME, start_time))
    model_path = os.path.join(MODEL_DIR, '{}_{}'.format(PROJECT_NAME, start_time))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph, max_queue=20)
        sess.run(init_op)

        try:
            for step in range(TOTAL_EPOCHES + 1):
                images, noises = batch_gen.__next__()
                feed_dict = {batch_images: images, batch_noises: noises}

                if step > 0 and step % 200 == 0:
                    _, _, d_loss_, g_loss_, summary_str, global_step_ = sess.run(
                        [d_train_op, g_train_op, d_loss, g_loss, summary_op, global_step], feed_dict)

                    print('Iter: {} | generator_loss: {:.3} | discriminator_loss: {:.3}'.format(step, g_loss_, d_loss_))
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    saver.save(sess, os.path.join(model_path, PROJECT_NAME + '.ckpt'), step)
                else:
                    sess.run([d_train_op, g_train_op, global_step], feed_dict)
        except KeyboardInterrupt:
            print('Interrupted by user!')
            saver.save(sess, os.path.join(model_path, PROJECT_NAME + '_interrupt_' + '.ckpt'), step)
        finally:
            print('done!')
            summary_writer.close()


if __name__ == '__main__':
    _main()