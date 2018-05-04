import os
import logging
import argparse
from copy import deepcopy

import numpy as np
import tensorflow as tf

from wordcnn import WordCNN

from utils.core import train, evaluate
from utils.misc import load_data, build_metric


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
    cfg = deepcopy(args)

    cfg.data = os.path.expanduser(cfg.data)

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
    env.model = m

    # we do not save the embedding here since embedding is not trained.
    env.saver = tf.train.Saver(var_list=m.varlist)

    env = build_metric(env, cfg)

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    return env


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
        cfg.data, cfg.bipolar)

    info('training model')
    train(env, X_train, y_train, X_valid, y_valid, load=False,
          batch_size=cfg.batch_size, epochs=cfg.epochs, name=cfg.name)
    evaluate(env, X_test, y_test, batch_size=cfg.batch_size)
    env.sess.close()


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
