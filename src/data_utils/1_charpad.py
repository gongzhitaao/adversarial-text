import os
import logging
import argparse
import re

from tqdm import tqdm


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Encode/decode each token to a appropriate form.')
    parser.add_argument('fname', metavar='FILE', type=str, help='file name')
    parser.add_argument('--seqlen', metavar='N', type=int,
                        help='maximum sequence length')
    parser.add_argument('--wordlen', metavar='N', type=int,
                        help='maximum word length')
    parser.add_argument('--ascii', dest='allascii', action='store_true',
                        help='replace nonascii characters with unk character')
    enc = parser.add_mutually_exclusive_group(required=True)
    enc.add_argument('--encode', dest='encode', action='store_true',
                     help='encode file if True, otherwise decode file.')
    enc.add_argument('--decode', dest='encode', action='store_false',
                     help='encode file if True, otherwise decode file.')
    parser.add_argument('--sow', type=str, default='{',
                        help='start of word symbol')
    parser.add_argument('--eow', type=str, default='}',
                        help='end of word symbol')
    parser.add_argument('--eos', type=str, default='+',
                        help='end of sentence symbol')
    parser.add_argument('--pad', type=str, default=' ', help='padding symbol')
    parser.add_argument('--unk', type=str, default='|', help='unknown char')
    parser.set_defaults(allascii=False)
    parser.set_defaults(encode=True)
    return parser.parse_args()


def encode_token(fname, args):
    info('reading and cleanup {}'.format(fname))
    # remove special symbols from the original text, append eos symbol
    lines = [re.sub(r'[{0}{1}{2}]'.format(args.sow, args.eow, args.eos),
                    args.unk, line.strip()) for line in open(fname, 'r')]
    if args.allascii:
        info('replacing non-ascii characters with "{}"'.format(args.unk))
        lines = [''.join(c if ord(c) < 128 else args.unk for c in line)
                 for line in lines]
    info('padding')
    maxlen = args.seqlen * (args.wordlen + 3) + 1
    for line in tqdm(lines):
        cur = line.split()[:args.seqlen]
        cur = ' '.join(['{word:{pad}<{maxlen}}'.format(
            # truncate tokens to maxlen, add sow and eow
            word='{sow}{token:.{maxlen}}{eow}'.format(
                sow=args.sow, token=t, maxlen=args.wordlen, eow=args.eow),
            pad=args.pad, maxlen=args.wordlen+2) for t in cur] + [args.eos])
        cur = '{sent:{pad}<{maxlen}}'.format(sent=cur, pad=args.pad,
                                             maxlen=maxlen)
        print(cur)


def decode_token(fname, args):
    info('reading lines')
    lines = [line for line in open(fname, 'r')]
    # split, discard the last token eos, and remove bow, eow.
    a, b = len(args.sow), len(args.eow)
    for line in tqdm(lines):
        cur = ' '.join([w.strip(args.pad)[a:-b] for w in line.split()[:-1]])
        print(cur)


def main(args):
    fn = encode_token if args.encode else decode_token
    fn(os.path.expanduser(args.fname), args)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
