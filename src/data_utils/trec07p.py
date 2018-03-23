import logging
import os
import email
import mmap
import random

import chardet
from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info

TREC07P = os.path.expanduser('~/data/trec07p')


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def extract_mail_body():
    def _extract(ifname, ofname, errfile, label):
        # First we detect the encoding with chardet
        with open(ifname, 'rb') as r:
            enc = chardet.detect(r.read())['encoding']

        try:
            # Now open the file with the detected encoding
            with open(ifname, 'r', encoding=enc) as r:
                raw_msg = r.read()
        except UnicodeDecodeError:
            with open(errfile, 'a') as w:
                w.write('{0} {1}\n'.format(label, ifname))
            return

        msg = email.message_from_string(raw_msg)
        body = ''
        for part in msg.walk():
            if part.get_content_type() in ['text/plain', 'text/html']:
                body += part.get_payload()

        with open(ofname, 'w', encoding='utf-8') as w:
            w.write(body.strip())

    basedir = TREC07P
    os.makedirs(os.path.join(basedir, 'spam'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'ham'), exist_ok=True)

    errfile = os.path.join(basedir, 'err.txt')
    with open(errfile, 'w') as w:
        w.write('')

    index = os.path.join(basedir, 'full/index')
    with open(index, 'r') as r:
        for line in tqdm(r, total=get_num_lines(index)):
            label, fname = line.split()
            basename = os.path.basename(fname.strip())
            ifname = os.path.join(basedir, 'data/{}'.format(basename))
            ofname = os.path.join(basedir, label, basename)
            _extract(ifname, ofname, errfile, label)


def merge_all():
    def _merge(d, fname):
        files = os.listdir(d)
        with open(fname, 'a') as w:
            for fname in tqdm(files):
                p = os.path.join(d, fname)
                with open(p, 'r') as r:
                    text = r.read()
                    text = ' '.join(text.split())
                w.write(text + '\n')

    basedir = TREC07P
    hamdir = os.path.join(basedir, 'ham')
    spamdir = os.path.join(basedir, 'spam')

    _merge(hamdir, os.path.join(basedir, 'ham.txt'))
    _merge(spamdir, os.path.join(basedir, 'spam.txt'))


def split_train_test(train_split=0.7):
    basedir = TREC07P
    hamfile = os.path.join(basedir, 'ham.txt')
    spamfile = os.path.join(basedir, 'spam.txt')

    hams = [l.strip() for l in open(hamfile, 'r')]
    spams = [l.strip() for l in open(spamfile, 'r')]

    info('hams {}'.format(len(hams)))
    info('spams {}'.format(len(spams)))

    # balance the two dataset
    N = min(len(hams), len(spams))
    random.shuffle(hams)
    random.shuffle(spams)
    hams = hams[:N]
    spams = spams[:N]

    def _save(fname, d):
        info('saving {}'.format(fname))
        with open(fname, 'w') as w:
            w.write('\n'.join(d))

    n_train = int(N * train_split)
    _save(os.path.join(basedir, 'train-0.txt'), hams[:n_train])
    _save(os.path.join(basedir, 'train-1.txt'), spams[:n_train])
    _save(os.path.join(basedir, 'test-0.txt'), hams[n_train:])
    _save(os.path.join(basedir, 'test-1.txt'), spams[n_train:])


if __name__ == '__main__':
    info('THE BEGIN')
    # extract_mail_body()
    # merge_all()
    split_train_test(train_split=0.7)
    info('THE END')
