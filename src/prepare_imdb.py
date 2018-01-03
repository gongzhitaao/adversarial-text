import os
from itertools import chain

import numpy as np

from gensim.models import KeyedVectors

import nltk

from utils import Timer, tick


# Number of recorders per category, i.e., N positive reviews for training, N
# negative reviews for training, same for testing.
N = 12500


@tick
def collect_reviews(inpath='~/data/aclImdb', outpath='~/data/imdb'):
    """
    Collect reviews into four files for convenience.

    No preprocessing is done on raw data.  The four files are train-pos.txt,
    train-neg.txt, test-pos.txt, and test-neg.txt.
    """
    inpath = os.path.expanduser(inpath)
    outpath = os.path.expanduser(outpath)

    if os.path.exists(os.path.join(outpath, 'train-pos.txt')):
        return

    for i in ['train', 'test']:
        for j in ['pos', 'neg']:
            curdir = os.path.join(inpath, i, j)
            outfile = os.path.join(outpath, '{0}-{1}.txt'.format(i, j))
            reviews = [None] * N

            print('\nReading {}'.format(curdir))
            for k, elm in enumerate(os.listdir(curdir)):
                with open(os.path.join(curdir, elm), 'r') as r:
                    s = r.read().strip()
                    reviews[k] = s

            print('\nSaving {}'.format(outfile))
            with open(outfile, 'w') as w:
                w.write('\n'.join(reviews))


@tick
def to_embeddings(w2v, fname, maxlen=400, dim=300):
    samples = [nltk.word_tokenize(line) for line in open(fname, 'r')]
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
    vecs = vecs.astype(np.float16)
    return vecs


@tick
def prepare_imdb(w2v, filepath='imdb/imdb.npz', rawpath='~/data/aclImdb'):
    filepath = os.path.expanduser('~/data/imdb/imdb.npz')
    datapath = os.path.expanduser('~/data/imdb')
    rawpath = os.path.expanduser(rawpath)

    print('\nGenerating training data')
    X_train_pos = to_embeddings(w2v, os.path.join(datapath, 'train-pos.txt'))
    X_train_neg = to_embeddings(w2v, os.path.join(datapath, 'train-neg.txt'))
    X_train = np.vstack((X_train_pos, X_train_neg))
    y_train = np.append(np.zeros(X_train_pos.shape[0], dtype=np.uint8),
                        np.ones(X_train_neg.shape[0], dtype=np.uint8))

    print('\nGenerating testing data')
    X_test_pos = to_embeddings(w2v, os.path.join(datapath, 'test-pos.txt'))
    X_test_neg = to_embeddings(w2v, os.path.join(datapath, 'test-neg.txt'))
    X_test = np.vstack((X_test_pos, X_test_neg))
    y_test = np.append(np.zeros(X_test_pos.shape[0], dtype=np.uint8),
                       np.ones(X_test_neg.shape[0], dtype=np.uint8))

    print('\nSaving {}'.format(filepath))
    np.savez(filepath, X_train=X_train, y_train=y_train, X_test=X_test,
             y_test=y_test)

    return X_train, y_train, X_test, y_test


collect_reviews()

with Timer('\nLoading word2vec'):
    ifname = os.path.expanduser('~/data/glove/glove.840B.300d.w2v')
    w2v = KeyedVectors.load(ifname)

prepare_imdb(w2v)
