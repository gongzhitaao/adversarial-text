"""
Convenient utilities, include training, evaluation and prediction.
"""
import os
import logging

import numpy as np
from tqdm import tqdm

from .ticktock import tick


__all__ = ['train', 'evaluate', 'predict']


logger = logging.getLogger(__name__)
info = logger.info


@tick
def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    assert hasattr(env, 'sess')

    if load:
        info('loading saved model')
        assert hasattr(env, 'saver')
        return env.saver.restore(env.sess, 'model/{}'.format(name))

    info('train model')

    assert hasattr(env, 'saver')
    assert hasattr(env, 'train_op')
    assert hasattr(env, 'training')
    assert hasattr(env, 'x')
    assert hasattr(env, 'y')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    for epoch in range(epochs):
        info('epoch {0}/{1}'.format(epoch+1, epochs))

        if shuffle:
            info('shuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        info('batch training')
        for batch in tqdm(range(n_batch), total=n_batch):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {env.x: X_data[start:end],
                         env.y: y_data[start:end],
                         env.training: True}
            env.sess.run(env.train_op, feed_dict=feed_dict)

        if X_valid is not None:
            evaluate(env, X_valid, y_valid, batch_size=batch_size)

    if hasattr(env, 'saver'):
        info('saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(env.sess, 'model/{}'.format(name))


@tick
def evaluate(env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    info('evaluating...')

    assert hasattr(env, 'sess')
    assert hasattr(env, 'x')
    assert hasattr(env, 'y')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    for batch in tqdm(range(n_batch), total=n_batch):
        end = min(n_sample, (batch+1) * batch_size)
        start = end - batch_size
        feed_dict = {env.x: X_data[start:end], env.y: y_data[start:end]}
        batch_loss, batch_acc = env.sess.run([env.loss, env.acc],
                                             feed_dict=feed_dict)
        loss += batch_loss * batch_size
        acc += batch_acc * batch_size
    loss /= n_batch * batch_size
    acc /= n_batch * batch_size

    info('loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


@tick
def predict(env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    info('predicting')

    assert hasattr(env, 'sess')
    assert hasattr(env, 'x')

    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in tqdm(range(n_batch), total=n_batch):
        end = min(n_sample, (batch+1) * batch_size)
        start = end - batch_size
        feed_dict = {env.x: X_data[start:end]}
        batch_y = env.sess.run(env.ybar, feed_dict=feed_dict)
        yval[start:end] = batch_y
    return yval
