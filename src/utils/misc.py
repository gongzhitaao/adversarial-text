import logging
import os

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
import numpy as np
import tensorflow as tf
from tqdm import tqdm


__all__ = ['load_data', 'build_metric', 'ReverseEmbedding', 'isadv', 'postfn']


logger = logging.getLogger(__name__)
info = logger.info


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)

    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


def load_data(data, bipolar, validation_split=0.1):
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

    info('X_train shape: {}'.format(X_train.shape))
    info('y_train shape: {}'.format(y_train.shape))
    info('X_test shape: {}'.format(X_test.shape))
    info('y_test shape: {}'.format(y_test.shape))

    if validation_split > 0:
        n = int(X_train.shape[0] * validation_split)
        X_valid = X_train[:n]
        X_train = X_train[n:]
        y_valid = y_train[:n]
        y_train = y_train[n:]
        return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)
    return (X_train, y_train), (X_test, y_test)


def build_metric(env, cfg):
    if cfg.output == tf.sigmoid:
        y = tf.to_float(env.y)
        with tf.variable_scope('acc'):
            t0 = tf.greater(env.ybar, 0.5)
            t1 = tf.greater(y, 0.5)
            count = tf.equal(t0, t1)
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y, logits=env.model.logits)
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
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y, logits=env.model.logits)
            env.loss = tf.reduce_mean(xent)
    else:
        raise ValueError('Unknown output function')
    return env


class ReverseEmbedding:
    def __init__(self, w2v_file, index_file=None):
        self.w2v = KeyedVectors.load(os.path.expanduser(w2v_file))
        self.w2v.init_sims()
        self.indexer = None
        if index_file is not None:
            self.indexer = AnnoyIndexer()
            self.indexer.load(os.path.expanduser(index_file))
            self.indexer.model = self.w2v

    def reverse_embedding(self, vec, unk='<unk>'):
        indices = np.empty(vec.shape[:-1], dtype=np.int32)
        sents = []
        with DisableLogger():
            for i, cur in tqdm(enumerate(vec), total=vec.shape[0]):
                tokens = [self.w2v.most_similar([v], topn=1,
                                                indexer=self.indexer)[0][0]
                          for v in cur]
                indices[i] = [self.w2v.vocab[w].index for w in tokens]
                sents.append(' '.join(tokens))
        return (indices, sents)


def isadv(*, y, y_pred, bipolar=False):
    if y_pred.shape[1] == 1:
        thres = 0. if bipolar else 0.5
        z = y_pred > thres
    else:
        z = np.argmax(y_pred, axis=1)
    return y.flatten() != z.flatten()


def postfn(cfg, ind, X_adv, X_sents, y_data, y_adv):
    advind = isadv(y=y_data, y_pred=y_adv, bipolar=cfg.bipolar)
    fname = os.path.join('out', cfg.outfile)
    dat = {}
    for i in range(cfg.n_classes):
        cur = np.where(np.all([advind, y_data.flatten() == i], axis=0))[0]
        dat['x{}'.format(i)] = ind[cur]
        sents = [X_sents[x] for x in cur]
        fn = '{}-{}.txt'.format(fname, i)
        info('saving {}'.format(fn))
        with open(fn, 'w') as w:
            w.write('\n'.join(sents))
    fn = '{}-ind.npz'.format(fname)
    info('saving {}'.format(fn))
    np.savez(fn, **dat)
