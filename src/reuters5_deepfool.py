import os
import logging

import numpy as np
import tensorflow as tf

from attacks import deepfool

from utils import train, evaluate, predict
from utils import model1b as model
from utils import metrics
from utils import load_data, make_deepfool, approx_knn, save_adv_sents


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


DATA = 'reuters5'
ADV = 'deepfool'
DATAPATH = '~/data/reuters/reuters5.npz'
BIPOLAR = None

if BIPOLAR is None:
    suffix = 'x'                # multi-label
elif BIPOLAR:
    suffix = 'b'                # bipolar, (-1, 1)
else:
    suffix = 'u'                # uni-polar, (0, 1)

MODEL = DATA + suffix
RESULT = '{0}_{1}'.format(DATA, ADV)

class Dummy:
    pass


env = Dummy()

info('constructing graph')

env.x = tf.placeholder(tf.float32, [None, 350, 300], 'x')
env.y = tf.placeholder(tf.float32, [None, 5], 'y')
env.training = tf.placeholder_with_default(False, (), 'mode')

env.ybar = model(env.x, training=env.training)
env.saver = tf.train.Saver()

env.loss, env.acc = metrics(y=env.y, ybar=env.ybar, bipolar=BIPOLAR)

optimizer = tf.train.AdamOptimizer()
env.train_op = optimizer.minimize(env.loss)

env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
env.noise = deepfool(model, env.x, epochs=env.adv_epochs, noise=True,
                     batch=True, clip_min=-10, clip_max=10)

info('initialize session')
env.sess = tf.InteractiveSession()
env.sess.run(tf.global_variables_initializer())
env.sess.run(tf.local_variables_initializer())

((X_train, y_train),
 (X_test, y_test),
 (X_valid, y_valid)) = load_data(DATAPATH, bipolar=BIPOLAR)

train(env, X_train, y_train, X_valid, y_valid, load=True, epochs=10,
      name=MODEL)

evaluate(env, X_test, y_test)

X_test = X_test[:128]
y_test = y_test[:128]

X_noise = make_deepfool(env, X_test, epochs=10)

X_adv = X_test + 15*X_noise
X_adv = np.clip(X_adv, -5, 5)

evaluate(env, X_adv, y_test)

X_quant, sents, dists = approx_knn(X_adv, X_test)

evaluate(env, X_quant, y_test)

save_adv_sents(sents, dists, y_test=y_test, y_clean=predict(env, X_test),
               y_adver=predict(env, X_quant), bipolar=BIPOLAR, name=RESULT)
