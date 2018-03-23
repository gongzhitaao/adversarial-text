import os
import logging
import argparse

from tqdm import tqdm
import numpy as np


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get each character from its UTF-8 encoding with chr.')
    parser.add_argument('fname', type=str, help='index file')
    return parser.parse_args()


def main(args):
    fn = os.path.expanduser(args.fname)
    info('loading {}'.format(fn))
    mat = np.load(fn).astype(np.int32)
    info('converting back to char')
    for line in mat:
        sent = ''.join(chr(c) for c in line)
        print(sent)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
