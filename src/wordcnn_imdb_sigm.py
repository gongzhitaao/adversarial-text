import os
import logging

import numpy as np
import tensorflow as tf

from wordcnn import WordCNN
from utils import train, evaluate
from utils import Timer


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

info('Constructing graph')

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

with Timer(msg='initialize session'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(),
             feed_dict={cfg.embedding: embedding})
    sess.run(tf.local_variables_initializer())
    env.sess = sess

with Timer(msg='loading data'):
    imdb = os.path.expanduser('~/data/imdb/imdb-word-seqlen-{}.npz'
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

load = False
train(env, X_train, y_train, X_valid, y_valid, load=load, epochs=10,
      name='imdb-word-sigm')
evaluate(env, X_test, y_test)

sess.close()
