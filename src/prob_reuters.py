import os

import numpy as np

from nltk.corpus import reuters

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import seaborn as sns

from utils import Timer


def foo(train, test):
    with Timer():
        trainlen = [0] * len(train)
        for i, cur in enumerate(train):
            sents = reuters.sents(cur)
            n = np.sum(len(sent) for sent in sents)
            trainlen[i] = n
        trainlen = np.array(trainlen)

        testlen = [0] * len(test)
        for i, cur in enumerate(test):
            sents = reuters.sents(cur)
            n = np.sum(len(sent) for sent in sents)
            testlen[i] = n

    print('mean: {0:.4f} std: {1:.4f}'
          .format(np.mean(trainlen), np.std(trainlen)))
    print('min: {0:.4f} max: {1:.4f}'
          .format(np.max(trainlen), np.min(trainlen)))

    print('mean: {0:.4f} std: {1:.4f}'
          .format(np.mean(testlen), np.std(testlen)))
    print('min: {0:.4f} max: {1:.4f}'
          .format(np.min(testlen), np.max(testlen)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(trainlen, ax=ax)
    sns.distplot(testlen, ax=ax)
    plt.tight_layout()
    os.makedirs('out', exist_ok=True)
    plt.savefig('out/reuters.pdf')

    val = np.sum(trainlen < 350) / len(trainlen)
    print(val)


labels = np.array(reuters.categories(), dtype='str')
fileids = [reuters.fileids(label) for label in labels]
sizes = np.array([len(ids) for ids in fileids])

ind = np.where(sizes > 1000)[0]
print(labels[ind])              # ['acq' 'earn']
print(sizes[ind])               # [2369 3964]

ind = np.where(np.logical_and(400 < sizes, sizes < 1000))[0]
print(labels[ind])
print(sizes[ind])

data = [fileids[i] for i in ind]

for d in data:
    train = [i for i in d if 'train' in i]
    test = [i for i in d if 'test' in i]
    print(len(train), len(test))
    assert len(train) + len(test) == len(d)

for d in data:
    train = [i for i in d if 'train' in i]
    test = [i for i in d if 'test' in i]
    foo(train, test)

# Reuters-2
# dataset   train  test
# 'acq'      1650   719
# 'earn'     2877  1087

# Reuter
# dataset     train  test
# 'crude'       389   189
# 'grain'       433   149
# 'interest'    347   131
# 'money-fx'    538   179
# 'trade'       368   117
