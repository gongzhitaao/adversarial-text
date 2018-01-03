import os
from itertools import chain

import numpy as np

from gensim.models import KeyedVectors

from nltk.corpus import reuters

from utils import Timer, tick


@tick
def to_embeddings(w2v, samples, maxlen, dim=300):
    N, L, D = len(samples), maxlen, dim
    print('Embedding: [{0}, {1}, {2}]'.format(N, L, D))
    vecs = np.zeros([N, L, D])
    for i, sent in enumerate(samples):
        sent = sent[:L]
        for j, w in enumerate(sent):
            try:
                v = w2v.word_vec(w)
            except KeyError:
                v = w2v.word_vec('<unk>')
            vecs[i, j] = v
    return vecs


@tick
def prepare_reuters2(w2v, pos, neg):
    n_train = 1650
    n_test = 719

    def _embed(ids):
        train = np.array([i for i in ids if 'train' in i], dtype=str)
        test = np.array([i for i in ids if 'test' in i], dtype=str)

        train = np.random.choice(train, size=n_train, replace=False)
        test = np.random.choice(test, size=n_test, replace=False)

        X_train = [list(chain.from_iterable(reuters.sents(i))) for i in train]
        X_train = to_embeddings(w2v, X_train, maxlen=160)
        X_test = [list(chain.from_iterable(reuters.sents(i))) for i in test]
        X_test = to_embeddings(w2v, X_test, maxlen=160)
        return X_train, X_test

    print('Embedding...')
    X_train_pos, X_test_pos = _embed(pos)
    X_train_neg, X_test_neg = _embed(neg)
    y_train_pos = np.zeros(n_train, dtype=np.uint8)
    y_test_pos = np.zeros(n_test, dtype=np.uint8)
    y_train_neg = np.ones(n_train, dtype=np.uint8)
    y_test_neg = np.ones(n_test, dtype=np.uint8)

    X_train = np.vstack((X_train_pos, X_train_neg)).astype(np.float16)
    X_test = np.vstack((X_test_pos, X_test_neg)).astype(np.float16)
    y_train = np.concatenate((y_train_pos, y_train_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))

    print('Saving data...')
    reuters2 = os.path.expanduser('~/data/reuters/reuters2.npz')
    np.savez(reuters2, X_train=X_train, y_train=y_train, X_test=X_test,
             y_test=y_test)


@tick
def prepare_reuters5(w2v, labels):
    n_train = 347
    n_test = 117

    def _embed(ids):
        train = np.array([i for i in ids if 'train' in i], dtype=str)
        test = np.array([i for i in ids if 'test' in i], dtype=str)

        train = np.random.choice(train, size=n_train, replace=False)
        test = np.random.choice(test, size=n_test, replace=False)

        X_train = [list(chain.from_iterable(reuters.sents(i))) for i in train]
        X_train = to_embeddings(w2v, X_train, maxlen=350)
        X_test = [list(chain.from_iterable(reuters.sents(i))) for i in test]
        X_test = to_embeddings(w2v, X_test, maxlen=350)
        return X_train, X_test

    print('Embedding...')
    n = len(labels)
    X_trains, X_tests = [None]*n, [None]*n
    y_trains, y_tests = [None]*n, [None]*n

    for i, label in enumerate(labels):
        X_trains[i], X_tests[i] = _embed(label)
        y_trains[i] = np.zeros(n_train, dtype=np.uint8) + i
        y_tests[i] = np.zeros(n_test, dtype=np.uint8) + i

    X_train = np.vstack(X_trains).astype(np.float16)
    X_test = np.vstack(X_tests).astype(np.float16)
    y_train = np.concatenate(y_trains)
    y_test = np.concatenate(y_tests)

    print('Saving data...')
    reuters5 = os.path.expanduser('~/data/reuters/reuters5.npz')
    np.savez(reuters5, X_train=X_train, y_train=y_train, X_test=X_test,
             y_test=y_test)


with Timer():
    print('Loading word2vec')
    ifname = os.path.expanduser('~/data/glove/glove.840B.300d.w2v')
    w2v = KeyedVectors.load(ifname)

pos = reuters.fileids('acq')
neg = reuters.fileids('earn')
prepare_reuters2(w2v, pos, neg)

labels = [reuters.fileids(label)
          for label in ['crude', 'grain', 'interest', 'money-fx', 'trade']]
prepare_reuters5(w2v, labels)
