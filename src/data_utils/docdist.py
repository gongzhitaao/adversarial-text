import os
import logging
import argparse
from functools import reduce
from pathlib import PurePath

import numpy as np
from gensim.models import KeyedVectors


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


class DisableLogger():
    def __enter__(self):
        logging.disable(100000)

    def __exit__(self, *_):
        logging.disable(logging.NOTSET)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Computer the distance among between sentences.')
    parser.add_argument('--origin', type=str)
    parser.add_argument('--adv', type=str)
    parser.add_argument('--w2v', metavar='FILE', type=str)
    return parser.parse_args()


def distance(w2v, sents0, sents1):
    return [[w2v.wmdistance(s0, s1),                              # wmd
             sum(x != y for x, y in zip(s0.split(), s1.split()))]  # n
            for s0, s1 in zip(sents0, sents1)]


def main(args):
    info('loading files')
    f0 = os.path.expanduser(args.origin)
    f1 = os.path.expanduser(args.adv)
    tmp = [line for line in open(f0)]
    p = PurePath(f1)
    # note we have a nasty hack here [:-6], the last six characters is
    # '-unpad', which is added when removing the <eos> and <pad> tokens.
    ind = np.load(str(p.parent / p.stem)[:-6] + '.npy')
    sents0 = [tmp[i] for i in ind]
    sents1 = [line for line in open(f1)]

    assert len(sents0) == len(sents1), '{0} {1}'.format(len(sents0), len(sents1))

    info('loading w2v')
    w2v = KeyedVectors.load(os.path.expanduser(args.w2v))
    with DisableLogger():
        dist = np.array(distance(w2v, sents0, sents1))
    avg = np.mean(dist, axis=0)
    std = np.std(dist, axis=0)
    print('WMD mean: {0:.4f} std: {1:.4f}'.format(avg[0], std[0]))
    print('N mean: {0:.4f} std: {1:.4f}'.format(avg[1], std[1]))


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
