import os
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.contrib import slim


def load_mnist(dataset_path):
    assert os.path.exists(dataset_path)
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
        batch_noise = np.random.standard_normal(batch_image.shape) / (2 * np.pi)
        yield batch_image, batch_noise


def gan_generator(noise_image, l2_weight, bn_params):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                            weights_regularizer=slim.l2_regularizer(l2_weight)):

            # 28 x 28 => 14 x 14
            conv1 = slim.repeat(noise_image, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=8, scope='conv1')
            pool1 = slim.avg_pool2d(conv1, [2, 2], scope='pool1')

            # 14 x 14 => 7 x 7
            conv2 = slim.repeat(pool1, 2, slim.conv2d, kernel_size=[3, 3], num_outputs=16, scope='conv2')
            pool2 = slim.avg_pool2d(conv2, [2, 2], scope='pool2')

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


def gan_discriminator(image, l2_weight, bn_params):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.leaky_relu,
                            weights_regularizer=slim.l2_regularizer(l2_weight)):
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3],
                                normalizer_fn=slim.batch_norm, normalizer_params=bn_params):
                # 28 x 28 => 14 x 14
                net = slim.repeat(image, 2, slim.conv2d, num_outputs=32, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                # 14 x 14 => 7 x 7
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=64, scope='conv2')

                # global average pooling
                net = tf.reduce_mean(net, axis=[1, 2], name='global_average_pool')

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, scope='fc3')
                net = slim.fully_connected(net, 10, scope='fc4')
                logits = slim.fully_connected(net, 1, activation_fn=tf.nn.tanh, scope='fc5')
    return logits


def build_loss(logits_real, logits_fake, smooth):
    discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_real, labels=tf.ones_like(logits_real, dtype=tf.float32) * smooth))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.zeros_like(logits_fake, dtype=tf.float32) + (1 - smooth)))
    discriminate_loss = discriminator_loss_real + discriminator_loss_fake

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.ones_like(logits_real, dtype=tf.float32)))

    return discriminate_loss, generator_loss


def network(batch_images, batch_noises, l2_weights, bn_params, smooth):
    logits_real = gan_discriminator(batch_images, l2_weights, bn_params)
    generate_images = gan_generator(batch_noises, l2_weights, bn_params)
    logits_fake = gan_discriminator(generate_images, l2_weights, bn_params)

    d_loss, g_loss = build_loss(logits_real, logits_fake, smooth)
    return d_loss, g_loss, generate_images


def display_batch_py(images, rows=8):
    n, h, w, c = images.shape
    images = images.reshape([rows, n // rows, h, w, c]).transpose(0, 2, 1, 3, 4).reshape([1, rows * h, n // rows * w, c])
    images = images * 127.5 + 127.5
    return images.astype(np.uint8)


def _main(train_args):
    with tf.name_scope('Inputs'):
        batch_images = tf.placeholder(dtype=tf.float32, shape=[None, train_args.image_size, train_args.image_size, 1], name='images')
        batch_noises = tf.placeholder(dtype=tf.float32, shape=[None, train_args.image_size, train_args.image_size, 1], name='noises')

    d_loss, g_loss, g_images = network(batch_images, batch_noises, train_args.l2_weight,
                                       bn_params={'decay': train_args.bn_decay, 'epsilon': train_args.bn_epsilon},
                                       smooth=train_args.smooth)

    with tf.name_scope('Summary'):
        tf.summary.scalar('generator_loss', g_loss)
        tf.summary.scalar('discriminator_loss', d_loss)

        assert train_args.batch_size // train_args.display_rows == train_args.batch_size / train_args.display_rows
        input_images = tf.py_func(display_batch_py, [batch_images, train_args.display_rows], tf.uint8)
        generate_images = tf.py_func(display_batch_py, [g_images, train_args.display_rows], tf.uint8)

        tf.summary.image('input_images', input_images)
        tf.summary.image('generate_images', generate_images)

    with tf.name_scope('Train'):
        train_vars = tf.trainable_variables()
        d_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in train_vars if var.name.startswith('generator')]

        global_step = tf.train.get_or_create_global_step()
        d_train_op = tf.train.GradientDescentOptimizer(train_args.learning_rate).minimize(d_loss, global_step=global_step, var_list=d_vars)
        g_train_op = tf.train.AdamOptimizer(train_args.learning_rate).minimize(g_loss, global_step=global_step, var_list=g_vars)
        train_op = tf.group(d_train_op, g_train_op)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()

    x_train, _, _, _ = load_mnist(train_args.dataset_path)
    batch_gen = batch_generator(x_train, batch_size=train_args.batch_size)
    saver = tf.train.Saver(max_to_keep=10)

    if not os.path.exists(train_args.log_dir):
        os.mkdir(train_args.log_dir)
    if not os.path.exists(train_args.model_dir):
        os.mkdir(train_args.model_dir)

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    log_path = os.path.join(train_args.log_dir, '{}_{}'.format(train_args.project_name, start_time))
    model_path = os.path.join(train_args.model_dir, '{}_{}'.format(train_args.project_name, start_time))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph, max_queue=20)
        sess.run(init_op)

        try:
            for step in range(train_args.epochs + 1):
                images, noises = batch_gen.__next__()
                feed_dict = {batch_images: images, batch_noises: noises}

                if step > 0 and step % train_args.refresh_interval == 0:
                    # _, _, d_loss_, g_loss_, summary_str, global_step_ = sess.run(
                    #     [d_train_op, g_train_op, d_loss, g_loss, summary_op, global_step], feed_dict)
                    _, d_loss_, g_loss_, summary_str, global_step_ = sess.run(
                        [train_op, d_loss, g_loss, summary_op, global_step], feed_dict)

                    print('Iter: {} | generator_loss: {:.3} | discriminator_loss: {:.3}'.format(step, g_loss_, d_loss_))
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    saver.save(sess, os.path.join(model_path, train_args.project_name + '.ckpt'), step)
                else:
                    # sess.run([d_train_op, g_train_op, global_step], feed_dict)
                    sess.run([train_op, global_step], feed_dict)
        except KeyboardInterrupt:
            print('Interrupted by user!')
            saver.save(sess, os.path.join(model_path, train_args.project_name + '_interrupt_' + '.ckpt'), step)
        finally:
            print('done!')
            summary_writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--project_name', help='name this project', type=str, default='gan_mnist')
    parser.add_argument('-l', '--learning_rate', help='set learning rate', type=float, default=0.01)
    parser.add_argument('-b', '--batch_size', help='training batch size', type=int, default=256)
    parser.add_argument('-e', '--epochs', help='training epochs', type=int, default=40000)
    parser.add_argument('-s', '--smooth', help='set smooth label', type=float, default=1.0)
    parser.add_argument('-r', '--refresh_interval', help='refresh log and save model interval', type=int, default=100)
    parser.add_argument('-d', '--dataset_path', help='mnist.npz path', type=str, default='./data/mnist.npz')

    parser.add_argument('--image_size', help='image size', type=int, default=28)
    parser.add_argument('--l2_weight', help='L2 regularization weight', type=float, default=0.0005)
    parser.add_argument('--bn_decay', help='batch normalization decay', type=float, default=0.995)
    parser.add_argument('--bn_epsilon', help='batch normalization epsilon', type=float, default=0.0001)

    parser.add_argument('--model_dir', help='save model directory', type=str, default='models')
    parser.add_argument('--log_dir', help='save log directory', type=str, default='logs')

    parser.add_argument('--display_rows', help='rows displayed in tensorboard image', type=int, default=16)
    return parser.parse_args()


if __name__ == '__main__':
    _main(get_args())
