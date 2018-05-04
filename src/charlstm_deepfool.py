import os
import logging
import argparse
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from charlstm import CharLSTM

from utils.core import train, evaluate, predict
from utils.misc import load_data, build_metric, index2char, postfn

from attacks import deepfool


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Attach CharLSTM.')

    parser.add_argument('--batch_size', metavar='N', type=int, default=64)
    parser.add_argument('--data', metavar='FILE', type=str, required=True)
    parser.add_argument('--drop_rate', metavar='N', type=float, default=0.2)
    parser.add_argument('--embedding_dim', metavar='FILE', type=str)
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

    parser.add_argument('--adv_epochs', metavar='N', type=int, default=5)
    parser.add_argument('--adv_eps', metavar='EPS', type=float, default=20)
    parser.add_argument('--outfile', metavar='FILE', type=str, required=True)
    parser.add_argument('--unk', metavar='UNK', type=str, default='|')

    ka = parser.add_mutually_exclusive_group()
    ka.add_argument('--keepall', dest='keepall', action='store_true',
                     help='save all generated texts.')
    ka.add_argument('--keepadv', dest='keepall', action='store_false',
                     help='save only adversarial texts.')
    parser.set_defaults(keepall=False)

    return parser.parse_args()


def config(args, embedding):
    cfg = deepcopy(args)
    cfg.data = os.path.expanduser(cfg.data)
    cfg.charlen = (cfg.seqlen * (cfg.wordlen
                                 + 2         # start/end of word symbol
                                 + 1)        # whitespace between tokens
                   + 1)                      # end of sentence symbol

    if cfg.n_classes > 2:
        cfg.output = tf.nn.softmax
    elif 2 == cfg.n_classes:
        cfg.output = tf.tanh if cfg.bipolar else tf.sigmoid

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
    env.ybar = m.predict(env.x, env.training)
    env.saver = tf.train.Saver()
    env = build_metric(env, cfg)

    with tf.variable_scope('deepfool'):
        env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
        env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
        xadv = deepfool(m, env.x, epochs=env.adv_epochs, eps=env.adv_eps,
                        batch=True, clip_min=-10, clip_max=10)
        env.xadv = m.reverse_embedding(xadv)
    return env


def make_adversarial(env, X_data):
    batch_size = env.cfg.batch_size
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in tqdm(range(n_batch), total=n_batch):
        end = min((batch + 1) * batch_size, n_sample)
        start = end - batch_size
        feed_dict = {env.x: X_data[start:end],
                     env.adv_epochs: env.cfg.adv_epochs,
                     env.adv_eps: env.cfg.adv_eps}
        xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
        X_adv[start:end] = xadv
    return X_adv


def main(args):
    info('loading embedding vec')
    embedding = np.eye(args.vocab_size).astype(np.float32)

    info('constructing config')
    cfg = config(args, embedding)

    info('constructing graph')
    env = build_graph(cfg)
    env.cfg = cfg

    info('initializing session')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(),
             feed_dict={cfg.embedding: embedding})
    sess.run(tf.local_variables_initializer())
    env.sess = sess

    info('loading data')
    (_, _), (X_data, y_data) = load_data(os.path.expanduser(cfg.data),
                                         cfg.bipolar, validation_split=-1)

    info('loading model')
    train(env, load=True, name=cfg.name)
    info('evaluating against clean test samples')
    evaluate(env, X_data, y_data, batch_size=cfg.batch_size)
    info('making adversarial texts')
    X_adv = make_adversarial(env, X_data)
    info('evaluating against adversarial texts')
    evaluate(env, X_adv, y_data, batch_size=cfg.batch_size)
    y_adv = predict(env, X_adv, batch_size=cfg.batch_size)
    env.sess.close()
    info('recover chars from indices')
    X_sents = index2char(X_adv, unk=cfg.unk)
    postfn(cfg, X_sents, y_data, y_adv)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
