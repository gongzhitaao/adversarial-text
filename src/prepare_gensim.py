import os

import numpy as np

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

from utils import Timer


with Timer() as t:
    ifname = os.path.expanduser('~/data/glove/glove.840B.300d.w2v.txt')
    ofname = os.path.expanduser('~/data/glove/glove.840B.300d.w2v')

    if False:
        print('Training model model')
        model = KeyedVectors.load_word2vec_format(ifname, binary=False)
        model.save(ofname)
    else:
        print('Loading model')
        model = KeyedVectors.load(ofname)
        print(model)

print.model

# print('Init sims')
# with Timer() as t:
#     model.init_sims()


# with Timer() as t:
#     indfile = os.path.expanduser('~/data/glove/glove.840B.300d.annoy')

#     if True:
#         print('Building index')
#         annoy_index = AnnoyIndexer(model, 100)
#     else:
#         print('Loading index')
#         annoy_index = AnnoyIndexer()
#         annoy_index.load(indfile)
#         annoy_index.model = model

# print('Querying with Annoy')
# with Timer() as t:
#     noise = np.random.random([300])
#     val = model.most_similar([noise], topn=3, indexer=annoy_index)
# print(len(val))

# print('Querying with gensim')
# with Timer() as t:
#     val = model.most_similar(noise, topn=3)
# print(val)
