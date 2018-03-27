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
    parser.add_argument('--embedding_dim', metavar='FILE', type=str)
    parser.add_argument('--epochs', metavar='N', type=int, default=10)
    parser.add_argument('--feature_maps', metavar='N1 [N2 N3 ...]', nargs='+',
                        default=[25, 50, 75, 100, 125, 150])
    parser.add_argument('--kernel_sizes', metavar='N1 [N2 N3 ...]', nargs='+',
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--highways', metavar='N', type=int, default=1)
    parser.add_argument('--lstm_units', metavar='N', type=int, default=256)
    parser.add_argument('--lstms', metavar='N', type=int, default=2)
    parser.add_argument('--n_classes', metavar='N', type=int, required=True)
    parser.add_argument('--name', metavar='MODEL', type=str)
    parser.add_argument('--seqlen', metavar='N', type=int, default=300)
    parser.add_argument('--wordlen', metavar='N', type=int, required=True)

    bip = parser.add_mutually_exclusive_group(required=True)
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

    cfg.drop_rate = args.drop_rate
    cfg.embedding_dim = args.embedding_dim
    cfg.feature_maps = args.feature_maps
    cfg.highways = args.highways
    cfg.kernel_sizes = args.kernel_sizes
    cfg.lstm_units = args.lstm_units
    cfg.lstms = args.lstms
    cfg.n_classes = args.n_classes
    cfg.seqlen = args.seqlen
    cfg.vocab_size = args.vocab_size
    cfg.wordlen = args.wordlen

    cfg.charlen = (cfg.seqlen * (cfg.wordlen
                                      + 2         # start/end of word symbol
                                      + 1)        # whitespace between tokens
                        + 1)                      # end of sentence symbol

    if args.n_classes > 2:
        cfg.output = tf.nn.softmax
    elif 2 == args.n_classes:
        cfg.output = tf.tanh if args.bipolar else tf.sigmoid

    cfg.embedding = tf.placeholder(tf.float32, embedding.shape)

    return cfg


def build_graph(cfg):
    class _Dummy:
        pass

    env = _Dummy()

    env.x = tf.placeholder(tf.int32, [cfg.batch_size, cfg.charlen], 'x')
    env.y = tf.placeholder(tf.float32, [cfg.batch_size, 1], 'y')
    env.training = tf.placeholder_with_default(False, (), 'mode')

    m = CharCNN(cfg)
    env.ybar = m.predict(env.x, env.training)
    env.savor = tf.train.Saver()

    with tf.variable_scope('acc'):
        t0 = tf.greater(env.ybar, 0.5)
        t1 = tf.greater(env.y, 0.5)
        count = tf.equal(t0, t1)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    with tf.variable_scope('loss'):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=env.y,
                                                       logits=m.logits)
        env.loss = tf.reduce_mean(xent)

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    return env


def load_data(data):
    d = np.load(data)
    X_train, y_train = d['X_train'], d['y_train']
    X_test, y_test = d['X_test'], d['y_test']

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

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
        os.path.expanduser(args.data))

    info('training model')
    train(env, X_train, y_train, X_valid, y_valid, load=False,
          batch_size=args.batch_size, epochs=args.epochs, name=args.name)
    evaluate(env, X_test, y_test, batch_size=args.batch_size)
    env.sess.close()


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
