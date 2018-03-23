import os
import csv
import logging
import random


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info

DATADIR = os.path.expanduser('~/data/fakenews')
RAW = os.path.join(DATADIR, 'fake_or_real_news.csv')
FAKE = os.path.join(DATADIR, 'fake.txt')
REAL = os.path.join(DATADIR, 'real.txt')


def parse_csv():
    info('parsing csv')
    with open(RAW, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        real, fake = [], []
        for row in reader:
            # print(row[''], row['title'], row['text'], row['label'])
            text = ' '.join(row['text'].strip().split())
            label = row['label'].strip().lower()
            if len(text) == 0:
                continue
            if 'real' == label:
                real.append(text)
            elif 'fake' == label:
                fake.append(text)

    info('# fake news: {}'.format(len(fake)))
    info('saving to {}'.format(FAKE))
    with open(FAKE, 'w') as w:
        w.write('\n'.join(fake))

    info('# real news: {}'.format(len(real)))
    info('saving to {}'.format(REAL))
    with open(REAL, 'w') as w:
        w.write('\n'.join(real))


def train_test_split(train_split):
    real = [line for line in open(REAL, 'r')]  # train-0
    fake = [line for line in open(FAKE, 'r')]  # train-1
    n_real, n_fake = len(real), len(fake)
    n_train = int(min(n_real, n_fake) * train_split)
    info('# training: {}'.format(n_train * 2))
    info('# test: {}'.format(n_real + n_fake - n_train*2))

    def _save(fname, dat):
        info('saving {}'.format(fname))
        with open(fname, 'w') as w:
            w.write('\n'.join(dat))

    random.shuffle(real)
    random.shuffle(fake)
    _save(os.path.join(DATADIR, 'train-0.txt'), real[:n_train])
    _save(os.path.join(DATADIR, 'train-1.txt'), fake[:n_train])
    _save(os.path.join(DATADIR, 'test-0.txt'), real[n_train:])
    _save(os.path.join(DATADIR, 'test-1.txt'), fake[n_train:])


if __name__ == '__main__':
    info('THE BEGIN')
    # parse_csv()
    train_test_split(train_split=0.7)
    info('THE END')
