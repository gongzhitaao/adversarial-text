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
        description='Computer the WMD among between sentences.')
    parser.add_argument('--origin', metavar='FILE', type=str)
    parser.add_argument('--others', metavar='F1 [F2 F3 ...]', type=str,
                        nargs='+')
    parser.add_argument('--outfile', metavar='FILE', type=str)
    parser.add_argument('--w2v', metavar='FILE', type=str)
    return parser.parse_args()


def find_common_sents(origin, generated):
    clean = [line for line in open(os.path.expanduser(origin), 'r')]
    contents = [clean]
    m, n = len(generated), len(clean)
    postions = np.empty((m, n), dtype=bool)
    indices = [list(range(n))]
    for i, f in enumerate(generated):
        f = os.path.expanduser(f)
        p = PurePath(f)
        ind = np.load(str(p.parent / p.stem) + '.npy')
        pos = np.zeros(n, dtype=bool)
        pos[ind] = True
        postions[i] = pos
        indices.append(sorted(ind))
        contents.append([line for line in open(f, 'r')])
    common = np.where(np.all(postions, axis=0))[0]
    info('total groups: {}'.format(len(common)))
    ret = [[] for _ in range(len(common))]
    for ind, sents in zip(indices, contents):
        ind2 = np.where(np.in1d(ind, common))[0]
        for i, j in enumerate(ind2):
            ret[i].append(sents[j])
    return ret


def distance(w2v, sents):
    origin = sents[0].split()
    return [(w2v.wmdistance(sents[0], sent),                    # wmd
             sum(x != y for x, y in zip(origin, sent.split())))  # n in n/L
            for sent in sents]


def main(args):
    sent_groups = find_common_sents(args.origin, args.others)
    info('loading w2v')
    w2v = KeyedVectors.load(os.path.expanduser(args.w2v))
    with DisableLogger():
        dist_groups = [distance(w2v, sents) for sents in sent_groups]
    fn = os.path.expanduser(args.outfile)
    info('saving {}'.format(fn))
    with open(fn, 'w') as w:
        for dists, sents in zip(dist_groups, sent_groups):
            for (wmd, n), sent in zip(dists, sents):
                w.write('{:.4f} {} {}'.format(wmd, n, sent))
            w.write('\n')


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
