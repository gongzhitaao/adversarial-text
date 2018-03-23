"""
Tokenize input text files.

The input text files are tokenized with nltk.word_tokenize(), optionally,
unescape HTML symbols, remove HTML tags.
"""
import os
import logging
import argparse
import html

import nltk
import bleach
from tqdm import tqdm


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='Tokenize input text files.')
    parser.add_argument('fname', metavar='FILE', type=str, help='file name')
    parser.add_argument('--unescape', dest='unescape', action='store_true',
                        help='unescape html symbols, e.g., &gt; -> >')
    parser.add_argument('--cleanup', dest='cleanup', action='store_true',
                        help='remove html tags')
    parser.set_defaults(unescape=False)
    parser.set_defaults(cleanup=False)
    return parser.parse_args()


def tokenize(fname, args):
    info('processing {}'.format(fname))
    lines = [line for line in open(fname, 'r') if len(line.strip()) > 0]
    if args.unescape:
        info('unescape HTML symbols')
        lines = [html.unescape(line) for line in tqdm(lines)]
    if args.cleanup:
        info('cleanup HTML tags')
        lines = [bleach.clean(line, tags=[], strip=True).strip()
                 for line in tqdm(lines)]
    if args.unescape and args.cleanup:
        # bleach.clean will by default escape the HTML symbols
        info('unescape HTML symbols again after bleach')
        lines = [html.unescape(line) for line in tqdm(lines)]
    info('tokenizing')
    for line in tqdm(lines):
        cur = nltk.word_tokenize(line)
        print(' '.join(cur).strip())


def main(args):
    tokenize(os.path.expanduser(args.fname), args)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
