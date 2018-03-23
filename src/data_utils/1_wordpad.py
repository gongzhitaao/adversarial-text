"""
Truncate/pad tokens to a fixed sequence length.
"""
import os
import logging
import re
import argparse

from tqdm import tqdm


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Truncate/pad tokens.')
    parser.add_argument('fname', type=str, help='file name')
    parser.add_argument('--seqlen', metavar='N', type=int)
    parser.add_argument('--eos', type=str, default='<eos>',
                        help='end of sentence symbol')
    parser.add_argument('--unk', type=str, default='<unk>',
                        help='unknown token symbol')
    parser.add_argument('--pad', type=str, default='<pad>',
                        help='padding symbol')
    return parser.parse_args()


def wordpad(fname, args):
    info('processing {}'.format(fname))
    # Replace any <eos> and <pad> appearing in the text with <unk>.
    lines = [re.sub(r'(({0})|({1}))'.format(args.eos, args.pad), args.unk,
                    line.strip()) for line in open(fname, 'r')]
    for line in tqdm(lines):
        cur = line.split()[:args.seqlen] + [args.eos]
        cur += [args.pad] * (args.seqlen + 1 - len(cur))
        print(' '.join(cur))


def main(args):
    wordpad(os.path.expanduser(args.fname), args)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
