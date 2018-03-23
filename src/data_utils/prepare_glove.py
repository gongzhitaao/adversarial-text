"""
By default, GloVe has <unk>, but no <pad>, <bos> (beginning of sentence),
<eos> (end of sentence) symbols.
"""

import os
import logging


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


DIM = 300

symbols = {'<pad>': [0]*DIM, '<bos>': [1]*DIM, '<eos>': [2]*DIM}

def get_num_lines(file_path):
    fp = open(file_path, 'r')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

for k, v in symbols.items():
