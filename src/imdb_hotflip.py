import os
import logging

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from charcnn import CharModel
from hotflip import hf_replace
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
            self.seqlen = 300
            self.wordlen = 20
            self.training = False
            self.prob_fn = tf.sigmoid
            self.drop_rate = 0.2
            self.batch_size = 20
            self.beam_width = 3


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

m = CharModel(cfg)
env.ybar = m.predict(env.x, env.training)
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

env.xadv = hf_replace(m, env.x, env.y, cfg.embedding_dim, dim,
                      beam_width=cfg.beam_width, chars=10)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
env.sess = sess

data = os.path.expanduser('~/data/imdb/imdb-char-seqlen-{0}-wordlen-{1}.npz'
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

load = True
train(env, X_train, y_train, X_valid, y_valid, load=load, epochs=10,
      batch_size=cfg.batch_size, name='imdb-char-sigm')
evaluate(env, X_test, y_test, batch_size=cfg.batch_size)


def make_hotflip(env, X_data, y_data, cfg):
    batch_size = cfg.batch_size
    W = cfg.beam_width
    L = cfg.seqlen * (cfg.wordlen + 3) + 1
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty((W, n_sample, L))

    for batch in tqdm(range(n_batch)):
        end = min(n_sample, (batch+1) * batch_size)
        start = end - batch_size
        feed_dict = {env.x: X_data[start:end], env.y: y_data[start:end]}
        xadv = sess.run(env.xadv, feed_dict=feed_dict)
        X_adv[:, start:end] = xadv
    return X_adv


X_adv = make_hotflip(env, X_test, y_test, cfg)
X_adv = np.reshape(X_adv, [-1, X_adv.shape[-1]])
labels = np.tile((y_test > 0.5).flatten(), cfg.beam_width).astype(np.int32)
tmp = np.reshape(labels, [-1, 1])
evaluate(env, X_adv, tmp, batch_size=cfg.batch_size)

for i in [0, 1]:
    cur = X_adv[labels == i]
    fn = 'out/imdb_char_hotflip-{}.npy'.format(i)
    info('saving {}'.format(fn))
    np.save(fn, cur)

info('THE END')
