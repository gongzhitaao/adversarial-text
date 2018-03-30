#!/bin/bash

seqlen=100

python run_wordcnn.py \
       --batch_size 64 \
       --data ~/data/reuters/reuters5/reuters5-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 10 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 5 \
       --name reuters5-word-seqlen-${seqlen} \
       --seqlen ${seqlen} \
       --units 128
