"""
TensorFlow implementation of character-level language model proposed in
Character-Aware Neural Language Models https://arxiv.org/abs/1508.06615
"""
import os
import logging

import numpy as np
import tensorflow as tf

from highway import Highway

from utils import train, evaluate


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


class Config:
    def __init__(self):
        # data
        self.data_dir = 'data'
        # model parameter
        self.embedding = None
        self.vocab_size = 128
        self.vec_size = 128
        self.n_classes = 2
        self.feature_maps = [25, 50, 75, 100, 125, 150]
        self.kernel_sizes = [1, 2, 3, 4, 5, 6]
        self.highways = 1
        self.lstm_units = 256
        self.lstms = 2
        self.wordlen = 20
        self.seqlen = 50
        # training
        self.dropout = 0.5
        self.hsm = 0
        self.learning_rate = 0.1
        self.learning_rate_decay = 0.5
        self.decay_when = 100
        self.param_init = 0.05
        self.batch_norm = False
        self.batch_size = 20
        self.epochs = 25
        self.grad_norm = 5
        # tokens
        self.eos = '+'          # end of sentence
        self.unk = '|'          # unknown word
        self.bow = '{'          # start of word
        self.eow = '}'          # end of word
        self.pad = ' '          # padding


class CharModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.build = False

    def _build(self):
        cfg = self.cfg
        if cfg.embedding is None:
            self.charvec = tf.get_variable('char_embedding', [cfg.vocab_size,
                                                              cfg.vec_size])
        else:
            self.charvec = tf.get_variable(
                name='char_embedding', shape=[cfg.vocab_size, cfg.vec_size],
                initializer=tf.constant_initializer(cfg.embedding),
                trainable=False)
        self.conv1ds = [tf.layers.Conv1D(filters, kernel_size, use_bias=True,
                                         activation=tf.tanh)
                        for filters, kernel_size in
                        zip(cfg.feature_maps, cfg.kernel_sizes)]
        self.highways = [Highway() for _ in range(cfg.highways)]
        self.cells = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(cfg.lstm_units)
            for _ in range(cfg.lstms)])
        self.resize = tf.layers.Dense(1)
        self.x_embed = None
        self.logits = None
        self.build = True

    def _add_inference_graph(self, x):
        self.x_embed = z = tf.nn.embedding_lookup(self.charvec, x)
        zs = [conv1d(z) for conv1d in self.conv1ds]
        zs = [tf.reduce_max(z, axis=1) for z in zs]  # max-over-time
        z = tf.concat(zs, axis=1)
        z = tf.expand_dims(z, axis=1)
        z, _ = tf.nn.dynamic_rnn(self.cells, z, dtype=tf.float32)
        z = tf.squeeze(z)
        self.logits = z = self.resize(z)
        return z

    def __call__(self, x):
        if not self.build:
            self._build()
        if self.logits is None:
            self._add_inference_graph(x)
        y = tf.argmax(self.logits, axis=1)
        return y

    def prob(self, x):
        if not self.build:
            self._build()
        if self.logits is None:
            self._add_inference_graph(x)
        return tf.sigmoid(self.logits)


cfg = Config()
cfg.embedding = np.eye(cfg.vocab_size)
m = CharModel(cfg)


class Dummy:
    pass


env = Dummy()

# Each word is padded to the same length, added with start/end of word symbol.
# Each sentence is padded to contain the same number of tokens, plus an end of
# sentence token.
dim = cfg.seqlen * (cfg.wordlen + 3) + 3

env.x = tf.placeholder(tf.int32, [cfg.batch_size, dim])
env.y = tf.placeholder(tf.float32, [cfg.batch_size, 1])
env.training = tf.placeholder_with_default(False, ())
env.ybar = m.prob(env.x)
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

data = os.path.expanduser(
    '~/data/reuters/reuters2/reuters2-seqlen-50-wordlen-20.npz')
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
      batch_size=cfg.batch_size, name='reuters2-char')
evaluate(env, X_test, y_test, batch_size=cfg.batch_size)
