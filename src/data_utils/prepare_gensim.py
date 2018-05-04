"""
Prepare Gensim word2vec model and annoy indexer for nearest neighbors.

Convert glove to word2vec format, according to
https://stackoverflow.com/a/41990999/1429714
"""
import os
import logging

import numpy as np

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)

    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


DIM = 300
glove_file = os.path.expanduser('~/data/glove/glove_tmp.txt')
w2v_file = os.path.expanduser('~/data/glove/glove.840B.300d.w2v.txt')
w2v_model = os.path.expanduser('~/data/glove/glove.840B.300d.w2v')
annoy_file = os.path.expanduser('~/data/glove/glove.840B.300d.annoy')


def extend_glove():
    symbols = {'<pad>': [1e-8]*DIM, '<bos>': [1]*DIM, '<eos>': [2]*DIM}
    with open(glove_file, 'a') as w:
        for k, v in symbols.items():
            w.write('{0} {1}\n'.format(k, ' '.join(str(e) for e in v)))


def build_word2vec():
    info('converting from glove to word2vec format')
    glove2word2vec(glove_file, w2v_file)
    info('training word2vec model')
    model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    model.save(w2v_model)


def build_annoy(w2v):
    info('building index')
    annoy_index = AnnoyIndexer(w2v, 500)
    info('saving index')
    annoy_index.save(annoy_file)


# extend_glove()
# build_word2vec()


info('loading model')
model = KeyedVectors.load(w2v_model)
info(model)

info('init sims')
model.init_sims()

# build_annoy(model)
info('loading annoy indexer')
annoy_index = AnnoyIndexer()
annoy_index.load(annoy_file)
annoy_index.model = model

noise = np.random.random([DIM])
noise = np.zeros(DIM)
info('querying with Annoy')
with DisableLogger():
    val = model.most_similar([noise, noise], topn=3, indexer=annoy_index)
info(val)

info('querying with gensim')
with DisableLogger():
    val = model.most_similar([noise, noise], topn=1)
info(val)
