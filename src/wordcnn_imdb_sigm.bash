#!/bin/bash

seqlen=300

python run_wordcnn.py \
       --batch_size 64 \
       --data ~/data/imdb/imdb-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 10 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name imdb-word-sigm-seqlen-${seqlen} \
       --seqlen ${seqlen} \
       --unipolar \
       --units 128
