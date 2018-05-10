import logging
import os

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
import numpy as np
import tensorflow as tf
from tqdm import tqdm


__all__ = ['load_data', 'build_metric', 'ReverseEmbedding', 'postfn',
           'index2char']


logger = logging.getLogger(__name__)
info = logger.info


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)

    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


def load_data(data, bipolar, validation_split=0.1):
    d = np.load(data)
    ret = []

    def _load(d, name):
        X_data, y_data = d['X_{}'.format(name)], d['y_{}'.format(name)]
        y_data = np.expand_dims(y_data, axis=1)
        if bipolar:
            y_data = 2 * y_data - 1
        return (X_data, y_data)

    if 'X_train' in d:
        X_train, y_train = _load(d, 'train')
        ret.append((X_train, y_train))
        info('X_train shape: {}'.format(X_train.shape))
        info('y_train shape: {}'.format(y_train.shape))

    if 'X_test' in d:
        X_test, y_test = _load(d, 'test')
        ret.append((X_test, y_test))
        info('X_test shape: {}'.format(X_test.shape))
        info('y_test shape: {}'.format(y_test.shape))

    if 'X_train' in locals() and validation_split > 0:
        ind = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[ind], y_train[ind]
        n = int(X_train.shape[0] * validation_split)
        X_valid = X_train[:n]
        X_train = X_train[n:]
        y_valid = y_train[:n]
        y_train = y_train[n:]
        ret[0] = (X_train, y_train)
        ret.append((X_valid, y_valid))

    if len(ret) > 1:
        return tuple(ret)
    return ret[0]


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
        info('loading w2v')
        self.w2v = KeyedVectors.load(os.path.expanduser(w2v_file))
        self.w2v.init_sims(replace=True)
        self.indexer = None
        if index_file is not None:
            info('loading annoy indexer')
            self.indexer = AnnoyIndexer()
            self.indexer.load(os.path.expanduser(index_file))
            self.indexer.model = self.w2v

    def reverse_embedding(self, vec, X_data):
        indices = np.empty(vec.shape[:-1], dtype=np.int32)
        sents = []
        pad = self.w2v.vocab['<pad>'].index
        eos = self.w2v.vocab['<eos>'].index
        with DisableLogger():
            # for i, (cur, dat) in tqdm(enumerate(zip(vec, X_data)),
            #                           total=vec.shape[0]):
            for i, (cur, dat) in enumerate(zip(vec, X_data)):
                # if the original token is <pad>, do not alter it
                tokens = []
                for v, x in zip(cur, dat):
                    if x == pad:
                        t = '<pad>'
                    elif x == eos:
                        t = '<eos>'
                    else:
                        t = self.w2v.most_similar([v], topn=1,
                                                  indexer=self.indexer)[0][0]
                    tokens.append(t)
                indices[i] = [self.w2v.vocab[w].index for w in tokens]
                sents.append(' '.join(tokens))
        return (indices, sents)

    def index(self, words, unk='<unk>'):
        indices = [self.w2v.vocab[w].index if w in self.w2v.vocab
                   else self.w2v.vocab[unk].index
                   for w in words]
        return np.array(indices)


def postfn(cfg, X_sents, y_data, y_adv):
    fname = os.path.join('out', cfg.outfile)

    y_data = y_data.flatten()
    if cfg.bipolar:
        y_data = (y_data + 1) // 2

    if cfg.keepall:
        isadv = np.ones(y_data.shape, dtype=bool)
    else:
        if y_adv.shape[1] == 1:
            thres = 0. if cfg.bipolar else 0.5
            z = y_adv > thres
        else:
            z = np.argmax(y_adv, axis=1)
        isadv = y_data != z.flatten()

    for i in range(cfg.n_classes):
        cur = np.where(np.all([isadv, y_data == i], axis=0))[0]
        # sents = ['{} '.format(z[x]) + X_sents[x] for x in cur]
        sents = [X_sents[x] for x in cur]
        fn = '{}-{}.txt'.format(fname, i)
        info('saving {}'.format(fn))
        with open(fn, 'w') as w:
            w.write('\n'.join(sents))
        if not cfg.keepall:
            fn = '{}-{}.npy'.format(fname, i)
            info('saving {}'.format(fn))
            np.save(fn, cur)


def index2char(mat, unk):
    sents = []
    for line in tqdm(mat, total=mat.shape[0]):
        sent = ''.join(unk if ord('\n') == c or ord('\r') == c else chr(c)
                       for c in line)
        sents.append(sent)
    return sents
