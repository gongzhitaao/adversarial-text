import os
import logging

import numpy as np
import tensorflow as tf

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

from tqdm import tqdm

from wordcnn import WordCNN
from utils import train, evaluate
from utils import Timer, tick

from attacks import fgm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


class Config:
    def __init__(self):
        self.embedding = None
        self.vocab_size = -1
        self.embedding_dim = 300
        self.n_classes = 2
        self.filters = 128
        self.kernel_size = 3
        self.units = 512
        self.seqlen = 300
        self.prob_fn = tf.sigmoid
        self.drop_rate = 0.2


class Dummy:
    pass


env = Dummy()

info('THE BEGIN')

cfg = Config()
embedding_file = os.path.expanduser(
    '~/data/glove/glove.840B.300d.w2v.vectors.npy')
embedding = np.load(embedding_file)
cfg.embedding = tf.placeholder(tf.float32, embedding.shape)
cfg.vocab_size = cfg.embedding.shape[0]

env.x = tf.placeholder(tf.int32, [None, cfg.seqlen + 1], 'x')
env.y = tf.placeholder(tf.float32, [None, 1], 'y')
env.training = tf.placeholder_with_default(False, (), 'mode')

m = WordCNN(cfg)
env.ybar = m.predict(env.x, env.training)

# we do not save the embedding here since embedding is not trained.
env.saver = tf.train.Saver(var_list=m.varlist)

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

with tf.variable_scope('model', reuse=True):
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.xadv = fgm(m, env.x, epochs=env.adv_epochs, eps=env.adv_eps,
                   clip_min=-10, clip_max=10)

with Timer(msg='initialize session'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(),
             feed_dict={cfg.embedding: embedding})
    sess.run(tf.local_variables_initializer())
    env.sess = sess

with Timer(msg='loading data'):
    imdb = os.path.expanduser(
        '~/data/reuters/reuters2/reuters2-word-seqlen-{}.npz'
        .format(cfg.seqlen))
    d = np.load(imdb)
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

load = True
train(env, X_train, y_train, X_valid, y_valid, load=load, epochs=5,
      name='reuters2-word-sigm')
evaluate(env, X_test, y_test)


def make_fgsm(env, X_data, epochs=1, eps=0.01, batch_size=128):
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    zadv = np.empty(list(X_data.shape)+[cfg.embedding_dim])
    for batch in tqdm(range(n_batch)):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_eps: eps,
                     env.adv_epochs: epochs}
        adv = env.sess.run(env.xadv, feed_dict=feed_dict)
        zadv[start:end] = adv
    return zadv

info('make fgsm')
zadv = make_fgsm(env, X_test, eps=0.2, epochs=1)


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)
    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


def reverse_embed(x_embed):
    info('loading word2vec model')
    fname = os.path.expanduser('~/data/glove/glove.840B.300d.w2v')
    w2v = KeyedVectors.load(fname)

    info('init w2v similarities')
    w2v.init_sims()

    info('loading index')
    fname = os.path.expanduser('~/data/glove/glove.840B.300d.annoy')
    annoy_index = AnnoyIndexer()
    annoy_index.load(fname)
    annoy_index.model = w2v

    X_adv = np.empty(x_embed.shape[:-1], dtype=np.int32)
    info('querying with Annoy')

    def _nearest(w2v, v, v0=np.zeros(300)):
        if np.allclose(v, v0):
            return '<pad>', v0
        with DisableLogger():
            w, _ = w2v.most_similar([v], topn=1, indexer=annoy_index)[0]
        v = w2v.word_vec(w)
        return w

    for i, sent in tqdm(enumerate(x_embed)):
        for j, v in enumerate(sent):
            w = _nearest(w2v, x_embed[i, j])
            ind = w2v.vocab[w].index
            X_adv[i, j] = ind
    return X_adv

info('reverse embedding')
X_adv = reverse_embed(zadv)
evaluate(env, X_adv, y_test)

info('THE END')
sess.close()
