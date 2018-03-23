import os
import logging

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from charcnn import CharModel
from utils import train, evaluate



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


class Config:
    def __init__(self):
            # model parameter
            self.embedding = None
            self.vocab_size = 128
            self.embedding_dim = 128
            self.n_classes = 2
            self.feature_maps = [25, 50, 75, 100, 125, 150]
            self.kernel_sizes = [1, 2, 3, 4, 5, 6]
            self.highways = 1
            self.lstm_units = 256
            self.lstms = 2
            self.seqlen = 50
            self.wordlen = 5
            self.training = False
            self.prob_fn = tf.sigmoid
            self.drop_rate = 0.2
            self.batch_size = 20


class Dummy:
    pass


env = Dummy()
cfg = Config()
cfg.embedding = np.eye(cfg.vocab_size).astype(np.float32)

dim = (cfg.seqlen * (cfg.wordlen
                     + 2         # start/end of word symbol
                     + 1)        # whitespace between tokens
       + 1)                      # end of sentence symbol

env.x = tf.placeholder(tf.int32, [cfg.batch_size, dim])
env.y = tf.placeholder(tf.float32, [cfg.batch_size, 1])
env.training = tf.placeholder_with_default(False, (), 'mode')
cfg.training = env.training

m = CharModel(cfg)
env.ybar = m.predict(env.x)
env.saver = tf.train.Saver()

with tf.variable_scope('loss'):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=env.y,
                                                   logits=m.logits)
    env.loss = tf.reduce_mean(xent)

with tf.variable_scope('acc'):
    t0 = tf.greater(env.ybar, 0.5)
    t1 = tf.greater(env.y, 0.5)
    count = tf.equal(t0, t1)
    env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

optimizer = tf.train.AdamOptimizer()
env.train_op = optimizer.minimize(env.loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
env.sess = sess

data = os.path.expanduser('~/data/fakenews/fakenews-char-seqlen-{0}-wordlen-{1}.npz'
                          .format(cfg.seqlen, cfg.wordlen))
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

load = False
train(env, X_train, y_train, X_valid, y_valid, load=load, epochs=5,
      batch_size=cfg.batch_size, name='imdb-fakenews-sigm')
evaluate(env, X_test, y_test, batch_size=cfg.batch_size)

info('THE END')
