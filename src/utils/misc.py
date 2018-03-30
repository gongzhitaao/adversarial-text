import logging
import os

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
import numpy as np
import tensorflow as tf
from tqdm import tqdm


__all__ = ['load_data', 'build_metric', 'ReverseEmbedding']


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
        sents = np.empty(vec.shape[:-1], dtype=np.int32)
        with DisableLogger():
            for i, cur in tqdm(enumerate(vec), total=vec.shape[0]):
                tokens = [self.w2v.most_similar([v], topn=1,
                                                indexer=self.indexer)[0][0]
                          for v in cur]
                sents[i] = [self.w2v.vocab[w].index
                            if w in self.w2v.vocab
                            else self.w2v.vocab[unk].index
                            for w in tokens]
        return sents
