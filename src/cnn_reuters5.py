import os

import numpy as np
import tensorflow as tf

from utils.core import train, evaluate
from utils import Timer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def model_fn(x, logits=False, training=False, name='ybar'):
    with tf.variable_scope('conv'):
        z = tf.layers.dropout(x, rate=0.2, training=training)
        z = tf.layers.conv1d(z, filters=256, kernel_size=3, padding='valid')
        z = tf.maximum(z, 0.2 * z)
        z = tf.reduce_max(z, axis=1, name='global_max_pooling')

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()[1:]
        z = tf.reshape(z, [-1, np.prod(shape)])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=512)
        z = tf.layers.dropout(z, rate=0.2, training=training)
        z = tf.maximum(z, 0.2 * z)

    logits_ = tf.layers.dense(z, units=5, name='logits')
    ybar = tf.nn.softmax(logits_, name=name)

    if logits:
        return ybar, logits_
    return ybar


class Dummy:
    pass


env = Dummy()

print('\nConstructing graph')

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, [None, 350, 300], 'x')
    env.y = tf.placeholder(tf.float32, [None, 5], 'y')
    env.training = tf.placeholder_with_default(False, (), 'mode')

    env.ybar, logits = model_fn(env.x, logits=True, training=env.training)

    env.saver = tf.train.Saver()

    with tf.name_scope('acc'):
        t0 = tf.greater(env.ybar, 0.5)
        t1 = tf.greater(env.y, 0.5)
        count = tf.equal(t0, t1)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.name_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.name_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)


print('\nInitialize session')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


reuters5 = os.path.expanduser('~/data/reuters/reuters5.npz')
d = np.load(reuters5)
X_train, y_train = d['X_train'], d['y_train']
X_test, y_test = d['X_test'], d['y_test']

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train).astype(np.float32)
y_test = to_categorical(y_test).astype(np.float32)

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * VALIDATION_SPLIT)
X_valid = X_train[:n]
X_train = X_train[n:]
y_valid = y_train[:n]
y_train = y_train[n:]

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))

load = True
train(sess, env, X_train, y_train, X_valid, y_valid, load=load, epochs=10,
      name='reuters5')
evaluate(sess, env, X_test, y_test)
