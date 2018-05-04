import os
import logging
import argparse

import numpy as np
import tensorflow as tf

from charlstm import CharLSTM

from utils.core import train, evaluate
from utils.misc import load_data, build_metric


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Classify text with CharLSTM')
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
    parser.add_argument('--vocab_size', metavar='N', type=int, default=128)
    parser.add_argument('--wordlen', metavar='N', type=int, required=True)

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

    cfg.batch_size = args.batch_size
    cfg.data = os.path.expanduser(args.data)
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
    env.y = tf.placeholder(tf.int32, [cfg.batch_size, 1], 'y')
    env.training = tf.placeholder_with_default(False, (), 'mode')

    m = CharLSTM(cfg)
    env.model = m
    env.ybar = m.predict(env.x, env.training)
    env.saver = tf.train.Saver()
    env = build_metric(env, cfg)

    return env


def main(args):
    info('loading embedding vec')
    embedding = np.eye(args.vocab_size).astype(np.float32)

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
    X_test, y_test = load_data(cfg.data, args.bipolar, -1)

    info('training model')
    train(env, load=True, name=args.name)
    evaluate(env, X_test, y_test, batch_size=args.batch_size)
    env.sess.close()


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
