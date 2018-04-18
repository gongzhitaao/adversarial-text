import os
import logging
import argparse

from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert each character to its UTF-8 encoding with ord.')
    parser.add_argument('--train', type=str, help='train.txt')
    parser.add_argument('--test', type=str, help='test.txt')
    parser.add_argument('--validation', type=str, help='validation.txt')
    parser.add_argument('--w2v', type=str, required=True,
                        help='path to saved KeyedVector model')
    parser.add_argument('--output', type=str, required=True,
                        help='output file name, without suffix')
    return parser.parse_args()


def token2index(fname, w2v):
    info('reading {}'.format(fname))
    with open(fname, 'r') as r:
        sents = r.read().splitlines()
    sents = [sent.split(maxsplit=1) for sent in sents]
    info('converting to indices')
    vec = [[int(sent[0])] +
           [w2v.vocab[w].index if w in w2v.vocab else w2v.vocab['<unk>'].index
            for w in sent[1].split()]
           for sent in tqdm(sents)]
    vec = np.array(vec)
    info('data shape {}'.format(vec.shape))
    X_data, y_data = vec[:, 1:], vec[:, 0]
    return X_data, y_data


def main(args):
    info('loading w2v')
    w2v = KeyedVectors.load(os.path.expanduser(args.w2v))
    data = {}
    if args.train is not None:
        info('generate training data')
        X_train, y_train = token2index(os.path.expanduser(args.train), w2v)
        data['X_train'], data['y_train'] = X_train, y_train
    if args.test is not None:
        info('generate test data')
        X_test, y_test = token2index(os.path.expanduser(args.test), w2v)
        data['X_test'], data['y_test'] = X_test, y_test
    if args.validation is not None:
        info('generate validation data')
        X_valid, y_valid = token2index(os.path.expanduser(args.validation),
                                       w2v)
        data['X_valid'], data['y_valid'] = X_valid, y_valid
    info('saving {}'.format(args.output))
    np.savez(os.path.expanduser(args.output), **data)


if __name__ == '__main__':
    info('THE BEGINNING')
    main(parse_args())
    info('THE END')
