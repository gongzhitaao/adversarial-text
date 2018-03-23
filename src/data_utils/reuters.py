import os
import logging
import re

from nltk.corpus import reuters

from utils import tick


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


@tick
def prepare_data(datadir, labels):
    def _extract(ids):
        X_train = [re.sub('[ \t\n]+', ' ', reuters.raw(i))
                   for i in ids if 'train' in i]
        X_test = [re.sub('[ \t\n]+', ' ', reuters.raw(i))
                  for i in ids if 'test' in i]
        return X_train, X_test

    for i, ids in enumerate(labels):
        info('processing {0}/{1}'.format(i+1, len(labels)))
        X_train, X_test = _extract(ids)
        fname = os.path.join(datadir, 'train-{}.txt'.format(i))
        with open(fname, 'w') as w:
            w.write('\n'.join(X_train))
        fname = os.path.join(datadir, 'test-{}.txt'.format(i))
        with open(fname, 'w') as w:
            w.write('\n'.join(X_test))


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/reuters')

    info('prepare reuters2')
    labels = [reuters.fileids(label) for label in ['acq', 'earn']]
    d = os.path.join(datadir, 'reuters2')
    os.makedirs(d, exist_ok=True)
    prepare_data(d, labels)

    info('prepare reuters5')
    labels = [reuters.fileids(label)
              for label in ['crude', 'grain', 'interest', 'money-fx', 'trade']]
    d = os.path.join(datadir, 'reuters5')
    os.makedirs(d, exist_ok=True)
    prepare_data(d, labels)
