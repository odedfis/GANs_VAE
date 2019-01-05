import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


def discriminator(img_in, reuse=None, keep_prob=0.8):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x


def generator(z, keep_prob=0.8, is_training=True):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data')
    tf.reset_default_graph()
    batch_size = 64
    n_noise = 64


    X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
    noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    g = generator(noise, keep_prob, is_training)
    d_real = discriminator(X_in)
    d_fake = discriminator(g, reuse=True)

    vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
    g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

    loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
    loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
    loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
    loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
        optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(60000):
        train_d = True
        train_g = True
        keep_prob_train = 0.6  # 0.5

        n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
        batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]

        d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d],
                                                    feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train,
                                                               is_training: True})

        d_real_ls = np.mean(d_real_ls)
        d_fake_ls = np.mean(d_fake_ls)
        g_ls = g_ls
        d_ls = d_ls

        if g_ls * 1.5 < d_ls:
            train_g = False
            pass
        if d_ls * 2 < g_ls:
            train_d = False
            pass

        if train_d:
            sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training: True})

        if train_g:
            sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training: True})
        if i % 200 == 0:
            saver.save(sess, './tensorflowModel.ckpt')
            tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)