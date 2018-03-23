import os
import re
import logging

from utils import tick


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


@tick
def prepare_imdb(datadir='~/data/imdb', rawdir='~/data/imdb/aclImdb'):
    """
    Collect reviews into four files for convenience.

    No preprocessing is done on raw data.  The four files are train-0.txt,
    train-1.txt, test-0.txt, and test-1.txt.
    """
    # Number of recorders per category, i.e., N positive reviews for training,
    # N negative reviews for training, same for testing.
    N = 12500

    datadir = os.path.expanduser(datadir)
    rawdir = os.path.expanduser(rawdir)

    if os.path.exists(os.path.join(datadir, 'train-0.txt')):
        return

    for i in ['train', 'test']:
        for j, k in enumerate(['pos', 'neg']):
            curdir = os.path.join(rawdir, i, k)
            outfile = os.path.join(datadir, '{0}-{1}.txt'.format(i, j))
            reviews = [''] * N

            info('reading {}'.format(curdir))
            for k, elm in enumerate(os.listdir(curdir)):
                with open(os.path.join(curdir, elm), 'r') as r:
                    s = re.sub('[ \t\n]+', ' ', r.read().strip())
                    reviews[k] = s

            info('saving {}'.format(outfile))
            with open(outfile, 'w') as w:
                w.write('\n'.join(reviews))


if __name__ == '__main__':
    prepare_imdb()
