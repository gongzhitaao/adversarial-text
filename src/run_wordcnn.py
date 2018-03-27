import os
import logging
import argparse

import numpy as np
import tensorflow as tf

from wordcnn import WordCNN
from utils import train, evaluate


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Classify text with WordCNN')
    parser.add_argument('--batch_size', metavar='N', type=int, default=64)
    parser.add_argument('--data', metavar='FILE', type=str, required=True)
    parser.add_argument('--drop_rate', metavar='N', type=float, default=0.2)
    parser.add_argument('--embedding', metavar='FILE', type=str)
    parser.add_argument('--epochs', metavar='N', type=int, default=10)
    parser.add_argument('--filters', metavar='N', type=int, default=128)
    parser.add_argument('--kernel_size', metavar='N', type=int, default=3)
    parser.add_argument('--n_classes', metavar='N', type=int, required=True)
    parser.add_argument('--name', metavar='MODEL', type=str)
    parser.add_argument('--seqlen', metavar='N', type=int, default=300)
    parser.add_argument('--units', metavar='N', type=int, default=512)

    bip = parser.add_mutually_exclusive_group()
    bip.add_argument('--bipolar', dest='bipolar', action='store_true',
                     help='-1/1 for output.')
    bip.add_argument('--unipolar', dest='bipolar', action='store_false',
                     help='0/1 for output.')
    parser.set_defaults(bipolar=False)

    return parser.parse_args()


def config(args, embedding):
    class _Dummy():
        pass
    cfg = _Dummy()

    cfg.n_classes = args.n_classes
    cfg.filters = args.filters
    cfg.kernel_size = args.kernel_size
    cfg.units = args.units
    cfg.seqlen = args.seqlen
    cfg.drop_rate = args.drop_rate

    if args.n_classes > 2:
        cfg.output = tf.nn.softmax
    elif 2 == args.n_classes:
        cfg.output = tf.tanh if args.bipolar else tf.sigmoid

    cfg.embedding = tf.placeholder(tf.float32, embedding.shape)
    cfg.vocab_size = embedding.shape[0]
    cfg.embedding_dim = embedding.shape[1]

    return cfg


def build_graph(cfg):
    class _Dummy:
        pass

    env = _Dummy()

    env.x = tf.placeholder(tf.int32, [None, cfg.seqlen + 1], 'x')
    env.y = tf.placeholder(tf.int32, [None, 1], 'y')
    env.training = tf.placeholder_with_default(False, (), 'mode')

    m = WordCNN(cfg)
    env.ybar = m.predict(env.x, env.training)

    # we do not save the embedding here since embedding is not trained.
    env.saver = tf.train.Saver(var_list=m.varlist)

    if cfg.output == tf.sigmoid:
        y = tf.to_float(env.y)
        with tf.variable_scope('acc'):
            t0 = tf.greater(env.ybar, 0.5)
            t1 = tf.greater(y, 0.5)
            count = tf.equal(t0, t1)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                           logits=m.logits)
            env.loss = tf.reduce_mean(xent)
    elif cfg.output == tf.tanh:
        y = tf.to_float(env.y)
        with tf.variable_scope('acc'):
            t0 = tf.greater(env.ybar, 0.0)
            t1 = tf.greater(y, 0.0)
            count = tf.equal(t0, t1)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            env.loss = tf.losses.mean_squared_error(
                labels=y, predictions=env.ybar,
                reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif cfg.output == tf.nn.softmax:
        y = tf.one_hot(env.y, cfg.n_classes, on_value=1.0, off_value=0.0)
        with tf.variable_scope('acc'):
            ybar = tf.argmax(env.ybar, axis=1, output_type=tf.int32)
            count = tf.equal(tf.reshape(env.y, [-1]), ybar)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                              logits=m.logits)
            env.loss = tf.reduce_mean(xent)
    else:
        raise ValueError('Unknown output function')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    return env


def load_data(data, bipolar):
    d = np.load(data)
    X_train, y_train = d['X_train'], d['y_train']
    X_test, y_test = d['X_test'], d['y_test']

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    if bipolar:
        y_train = 2 * y_train - 1
        y_test = 2 * y_test - 1

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * VALIDATION_SPLIT)
    X_valid = X_train[:n]
    X_train = X_train[n:]
    y_valid = y_train[:n]
    y_train = y_train[n:]

    info('X_train shape: {}'.format(X_train.shape))
    info('y_train shape: {}'.format(y_train.shape))
    info('X_test shape: {}'.format(X_test.shape))
    info('y_test shape: {}'.format(y_test.shape))

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)


def main(args):
    info('loading embedding vec')
    embedding = np.load(os.path.expanduser(args.embedding))

    info('constructing config')
    cfg = config(args, embedding)

    info('constructing graph')
    env = build_graph(cfg)

    info('initializing session')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(),
             feed_dict={cfg.embedding: embedding})
    sess.run(tf.local_variables_initializer())
    env.sess = sess

    info('loading data')
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_data(
        os.path.expanduser(args.data), args.bipolar)

    info('training model')
    train(env, X_train, y_train, X_valid, y_valid, load=False,
          batch_size=args.batch_size, epochs=args.epochs, name=args.name)
    evaluate(env, X_test, y_test, batch_size=args.batch_size)
    env.sess.close()


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
